"""Obstacle-size evaluation sharing carla_test.py's NoMaD inference and controller."""
import argparse
import faulthandler
import os
from datetime import datetime
from collections import deque
import hashlib
import json
import math
from pathlib import Path
from queue import Empty
import time
import traceback
import carla
import cv2
import numpy as np
import pygame
import torch
from torch.multiprocessing import Process, Queue, set_start_method
from general_navigation import carla_test as base
from general_navigation import depth_occupancy as dpt


_stream={'jpg':None}


def start_stream(port):
    """MJPEG live view of the annotated frames (what navigation.mp4 records).

    Open http://localhost:<port>/ (use ssh -L to forward it from the server).
    """
    import logging, threading
    from flask import Flask, Response
    app=Flask(__name__)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    @app.route('/')
    def index():
        return ('<html><head><title>CARLA obstacle test</title></head>'
                '<body style="background:#222;margin:0;text-align:center">'
                '<img src="/video_feed" style="width:96vw;image-rendering:pixelated"></body></html>')

    @app.route('/video_feed')
    def video_feed():
        def generate():
            while True:
                jpg=_stream['jpg']
                if jpg is None:
                    blank=np.zeros((128,256,3),np.uint8)
                    cv2.putText(blank,'Waiting for trial...',(20,64),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                    jpg=cv2.imencode('.jpg',blank)[1].tobytes()
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpg+b'\r\n'
                time.sleep(.05)
        return Response(generate(),mimetype='multipart/x-mixed-replace; boundary=frame')

    threading.Thread(target=lambda:app.run(host='127.0.0.1',port=port,threaded=True),daemon=True).start()
    print(f'Live stream: http://localhost:{port}/',flush=True)


def publish_frame(image):
    ok,buf=cv2.imencode('.jpg',image,[int(cv2.IMWRITE_JPEG_QUALITY),90])
    if ok:_stream['jpg']=buf.tobytes()


def pose_json(t):
    return [t.location.x,t.location.y,t.location.z,t.rotation.pitch,t.rotation.yaw,t.rotation.roll]


def reset(client,t):
    vehicle = client.world.player
    base.apply_vehicle_controls(client,base.stopped_controls())
    # Teleport with physics disabled so contact impulses from the previous
    # location cannot launch the vehicle at the new start/goal.
    vehicle.set_simulate_physics(False)
    vehicle.set_transform(t)
    client.sim_world.tick()
    vehicle.set_simulate_physics(True)
    vehicle.set_target_velocity(carla.Vector3D())
    vehicle.set_target_angular_velocity(carla.Vector3D())
    client.world.camera_manager.frame_packet=None
    client.world.camera_manager.surface=None
    for _ in range(30):
        base.apply_vehicle_controls(client,base.stopped_controls())
        if not client.game_loop():
            raise RuntimeError('Window closed during reset')
    actual = vehicle.get_transform()
    if (math.hypot(actual.location.x-t.location.x, actual.location.y-t.location.y)>1.0
            or abs(actual.location.z-t.location.z)>2.0
            or abs((actual.rotation.pitch+180)%360-180)>15
            or abs((actual.rotation.roll+180)%360-180)>15):
        raise RuntimeError(f'Unstable reset: requested={pose_json(t)}, actual={pose_json(actual)}')
    if base.camera_state(client) is None:raise RuntimeError('No camera image')


def spawn_obstacle(client, blueprint, waypoint):
    road = waypoint.transform
    # Spawn clear of the road collider, then lower the stationary actor using
    # its actual bounding box. Keep road height separate from spawn height.
    elevated = carla.Transform(
        carla.Location(x=road.location.x, y=road.location.y, z=road.location.z+3.0),
        road.rotation)
    obstacle = client.sim_world.try_spawn_actor(blueprint, elevated)
    if obstacle is None:
        nearby = [dict(id=a.id, type=a.type_id, pose=pose_json(a.get_transform()))
                  for a in client.sim_world.get_actors()
                  if a.type_id.startswith(('vehicle.', 'walker.', 'static.prop.'))
                  and a.get_location().distance(road.location)<10]
        raise RuntimeError(f'Cannot spawn {blueprint.id} at {pose_json(elevated)}; nearby actors={nearby}')
    try:
        obstacle.set_simulate_physics(False)
        box = obstacle.bounding_box
        grounded = carla.Transform(
            carla.Location(x=road.location.x, y=road.location.y,
                           z=road.location.z+box.extent.z-box.location.z+.03),
            road.rotation)
        obstacle.set_transform(grounded)
        client.sim_world.tick()
        return obstacle, grounded
    except Exception:
        obstacle.destroy()
        raise


def navigation_outcome(distance, goal_radius, motion, now, pose, active,
                       stuck_seconds, stuck_distance, goal_overshoot=0., fail_distance=5.):
    """Detect sustained lack of motion after inference starts, using simulator time."""
    if distance <= goal_radius:
        return 'arrived'
    if goal_overshoot >= fail_distance:
        return 'fail'
    if not active:
        motion.clear()
        return None
    motion.append((now, float(pose[0]), float(pose[1])))
    while len(motion)>1 and motion[1][0]<=now-stuck_seconds:
        motion.popleft()
    if motion and now-motion[0][0]>=stuck_seconds:
        # Extent of the entire window prevents a return to the same position
        # after a loop from being mistaken for being stationary.
        xs=[p[1] for p in motion];ys=[p[2] for p in motion]
        if math.hypot(max(xs)-min(xs),max(ys)-min(ys))<stuck_distance:
            return 'stuck'
    return None


def trial(client,model,config,device,start,goal,goal_path,label,bp,obstacle_wp,args):
    output=Path(args.output)/label;output.mkdir(exist_ok=False)
    obstacle=sensor=worker=writer=depth=None
    states=[];predictions=[];collisions=[];queues=[]
    last_image=None;motion=deque()
    report=dict(trial=label,outcome='error',obstacle=None,states=states,
        predictions=predictions,collisions=collisions,goal_pose=pose_json(goal),
        config=config,device=device,seed=base.DIFFUSION_SEED,num_samples=base.NUM_SAMPLES,
        path_selection=args.path_selection,goal_radius_m=args.goal_radius,
        video_frame_count=0,video_fps=1/base.CONTROL_DT,video_frames=[],
        fail_distance_m=args.fail_distance,
        video_terminal_frames=0,stuck_seconds=args.stuck_seconds,stuck_distance_m=args.stuck_distance,
        video_timing='One frame per control iteration; nominal 10 simulator fps')
    try:
        reset(client,start)
        report['initial_vehicle_pose']=pose_json(client.world.player.get_transform())
        if bp:
            obstacle,t=spawn_obstacle(client,client.sim_world.get_blueprint_library().find(bp),obstacle_wp)
            box=obstacle.bounding_box
            report['obstacle']=dict(blueprint=bp,length_m=2*box.extent.x,
                width_m=2*box.extent.y,height_m=2*box.extent.z,pose=pose_json(t),
                note='Real vehicle collision geometry; shape varies along with size')
        sensor=client.sim_world.spawn_actor(client.sim_world.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),attach_to=client.world.player)
        def collision(e):
            collisions.append(dict(frame=e.frame,other_type=e.other_actor.type_id,
                obstacle_collision=obstacle is not None and e.other_actor.id==obstacle.id))
        sensor.listen(collision)
        # RGB-D: ground-truth depth from a depth camera co-located with the RGB
        # camera (replaces Depth Anything V2). Logging only; control unchanged.
        grid=None;grid_cfg=None;depth_img=None;inflate_radius=None
        if args.depth_camera: #trial 開始時，設定 grid 參數並掛上深度相機
            grid_cfg=dpt.GridConfig(resolution=args.grid_resolution,h_min=args.grid_h_min,
                h_max=args.grid_h_max,max_range=args.grid_max_range)
            depth=dpt.DepthCamera(client)
            inflate_radius=depth.ego_box[2]+args.safety_margin
            report['depth_camera']=dict(depth.metadata(grid_cfg),inflate_radius_m=inflate_radius,
                safety_margin_m=args.safety_margin)
            if args.save_depth:(output/'depth').mkdir()
        iq,oq=Queue(maxsize=1),Queue(maxsize=1);queues=[iq,oq]
        worker=Process(target=base.async_nomad,args=(iq,oq,model,device,str(goal_path),config));worker.start()
        context=base.FixedImageContext(config['context_size']+1)
        history={};pending=False;path=samples_world=None;steer=0.;closest=float('inf')
        started=client.sim_world.get_snapshot().timestamp.elapsed_seconds
        wall_start=last_prediction=time.monotonic();watchdog=2.
        geometry=base.vehicle_geometry(client.world.player)
        while True:
            wall_tick=time.monotonic()
            if not client.game_loop():report['outcome']='window_closed';break
            snap=client.sim_world.get_snapshot();pose=base.vehicle_pose(client.world.player)
            history[snap.frame]=pose;history={f:p for f,p in history.items() if f>=snap.frame-32}
            packet=client.world.camera_manager.frame_packet
            if packet is not None and packet[0] in history:
                frame_id,timestamp,surface=packet
                rgb=pygame.surfarray.array3d(surface).transpose(1,0,2)
                context.add(timestamp,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),history[frame_id])
            if depth is not None: # 取最新的深度圖，投影成 occupancy grid
                got=depth.latest(history)
                if got is not None and (grid is None or got[0]!=grid.frame):
                    depth_frame,depth_img=got
                    grid=dpt.build_occupancy(depth_img,depth.K,depth.R,depth.t,grid_cfg,
                        frame=depth_frame,pose=history[depth_frame])
                    if args.save_depth:
                        np.savez_compressed(output/'depth'/f'{depth_frame:08d}.npz',
                            depth_m=depth_img.astype(np.float16),occupied=grid.occupied,
                            observed=grid.observed,pose=np.asarray(grid.pose))
            request=context.request()
            if not pending and request is not None:
                images,times,observation_pose=request
                iq.put_nowait((images,times,time.monotonic(),observation_pose));pending=True
            try:
                result,submitted,seconds,observation_pose,samples=oq.get_nowait()
                pending=False;last_prediction=time.monotonic();watchdog=max(2.,3*seconds)
                predictions.append(dict(sim_time=snap.timestamp.elapsed_seconds,
                    inference_seconds=seconds,observation_age_wall_s=last_prediction-submitted,
                    candidates=samples,observation_pose=observation_pose))
                if grid is not None and samples is not None:
                    # Candidates live in the observation frame; the grid in the
                    # depth frame's vehicle frame. Go through world coordinates.
                    inflated=grid.inflated(inflate_radius);checks=[]
                    for cand in samples: # NoMaD 每次產生候選時：膨脹障礙物，逐條檢查 8 條候選會不會撞 ###########################
                        local=base.world_to_local(base.local_to_world(cand,observation_pose),grid.pose)
                        local=dpt.extend_path(local,args.check_horizon)
                        checks.append(grid.path_clearance(local,inflated))
                    predictions[-1].update(grid_frame=grid.frame,
                        candidate_collides=[c[0] for c in checks],
                        candidate_clearance_m=[None if math.isinf(c[1]) else c[1] for c in checks],
                        feasible_count=sum(not c[0] for c in checks))
                if result['speed']>0 and samples is not None:
                    samples_world=np.asarray([base.local_to_world(p,observation_pose) for p in samples])
                    path=base.select_trajectory(samples,observation_pose,path,
                        policy='official_first' if args.path_selection=='official' else 'consensus')
                else:path=samples_world=None
            except Empty:pass
            controls=base.stopped_controls()
            if path is not None and time.monotonic()-last_prediction<=watchdog:
                controls,_=base.tracking_controls(base.world_to_local(path,pose),*geometry,steer,base.CONTROL_DT)
            steer=controls.steer
            distance=client.world.player.get_location().distance(goal.location)
            closest=min(distance,closest)
            # Signed distance past the goal plane along the goal's road heading.
            # Being far from the goal at the start must not count as overshoot.
            goal_yaw=math.radians(goal.rotation.yaw)
            overshoot=((pose[0]-goal.location.x)*math.cos(goal_yaw)
                       +(pose[1]-goal.location.y)*math.sin(goal_yaw))
            outcome=navigation_outcome(distance,args.goal_radius,motion,
                snap.timestamp.elapsed_seconds,pose,bool(predictions),
                args.stuck_seconds,args.stuck_distance,overshoot,args.fail_distance)
            if outcome:
                report['outcome']=outcome
                if outcome=='fail':
                    report['failure_reason']='goal_overshoot' # 車子沒有進入目標半徑，卻沿著道路方向開過了目標點
                if outcome=='stuck':
                    report['stuck_reason']='obstacle_contact' if any(c['obstacle_collision'] for c in collisions) else 'no_motion'
            elif ((args.timeout_sim>0 and snap.timestamp.elapsed_seconds-started>=args.timeout_sim)
                  or (args.timeout_wall>0 and time.monotonic()-wall_start>=args.timeout_wall)):
                report['outcome']='timeout'
                outcome='timeout' 
            states.append(dict(sim_time=snap.timestamp.elapsed_seconds,
                elapsed_sim_s=snap.timestamp.elapsed_seconds-started,pose=list(pose),
                vehicle_pose_3d=pose_json(client.world.player.get_transform()),
                goal_distance_m=distance,goal_overshoot_m=overshoot,speed_command=controls.speed,steer=steer))
            if grid is not None: # 每一步計算車頭前方的淨空距離
                clearance=grid.front_clearance(depth.ego_box[1],depth.ego_box[2])
                states[-1].update(depth_frame=grid.frame,
                    front_clearance_m=None if math.isinf(clearance) else clearance)
            state=base.camera_state(client)
            if state is not None:
                image=base.draw_nomad_paths(state.image.cv_image(),samples_world,path,pose)
                cv2.putText(image,f'{label} t={states[-1]["elapsed_sim_s"]:.1f} d={distance:.2f}m',(4,14),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,255,0),1)
                cv2.putText(image,(outcome or 'navigating').upper(),(4,34),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0) if outcome=='arrived' else (0,200,255),1)
                if depth is not None: # 影片中的深度圖和 BEV 畫面	
                    h=image.shape[0]
                    dimg=(dpt.colorize_depth(depth_img) if depth_img is not None
                          else np.zeros((h,image.shape[1],3),np.uint8))
                    dimg=cv2.resize(dimg,(round(dimg.shape[1]*h/dimg.shape[0]),h),interpolation=cv2.INTER_NEAREST)
                    to_grid=(lambda paths:None if paths is None or grid is None else
                             [base.world_to_local(p,grid.pose) for p in paths])
                    sel=to_grid(None if path is None else [path])
                    feas=(None if not predictions or 'candidate_collides' not in predictions[-1]
                          else [not c for c in predictions[-1]['candidate_collides']])
                    cands=to_grid(samples_world)
                    if cands is not None:cands=[dpt.extend_path(c,args.check_horizon) for c in cands]
                    bev=dpt.render_bev(grid,(h,h),candidates=cands,
                        selected=None if sel is None else sel[0],
                        feasible=feas if cands is not None and feas is not None and len(feas)==len(cands) else None,
                        ego_box=depth.ego_box,inflate_radius=inflate_radius)
                    image=np.hstack((image,dimg,bev))
                if args.stream_port:publish_frame(image)
                last_image=image.copy()
                if writer is None:
                    writer=cv2.VideoWriter(str(output/'navigation.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1/base.CONTROL_DT,(image.shape[1],image.shape[0]))
                    if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
                writer.write(image)
                report['video_frame_count']+=1
                report['video_frames'].append(dict(index=report['video_frame_count']-1,
                    sim_time=snap.timestamp.elapsed_seconds))
                if report['video_frame_count']==1:cv2.imwrite(str(output/'initial.png'),image)
            else:
                raise RuntimeError('Camera image missing during navigation; video would be incomplete')
            if len(states)%50==0:
                (output/'progress.json').write_text(json.dumps(dict(steps=len(states),last_state=states[-1],prediction_count=len(predictions))))
            if outcome:report['outcome']=outcome;break
            if not worker.is_alive():raise RuntimeError('Inference worker exited')
            base.apply_vehicle_controls(client,controls)
            time.sleep(max(0,base.CONTROL_DT-(time.monotonic()-wall_tick)))
    except Exception:
        report['error']=traceback.format_exc();print(report['error'],flush=True)
    finally:
        # Finalize the MP4 before any simulator RPC that might hang after a crash.
        if writer:
            if last_image is not None:
                cv2.rectangle(last_image,(0,19),(last_image.shape[1],44),(0,0,0),-1)
                cv2.putText(last_image,report['outcome'].upper(),(4,36),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0) if report['outcome']=='arrived' else (0,200,255),1)
                cv2.imwrite(str(output/'last.png'),last_image)
                if args.stream_port:publish_frame(last_image)
                for _ in range(round(2/base.CONTROL_DT)):
                    writer.write(last_image)
                    report['video_terminal_frames']+=1
                    report['video_frame_count']+=1
            writer.release()
            report['video_path']='navigation.mp4'
            report['video_duration_s']=report['video_frame_count']*base.CONTROL_DT
            capture=cv2.VideoCapture(str(output/'navigation.mp4'))
            report['video_saved_frame_count']=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            report['video_covers_recorded_steps']=(report['video_saved_frame_count']==report['video_frame_count'] and len(states)+report['video_terminal_frames']==report['video_frame_count'])
        (output/'report.json').write_text(json.dumps(report,indent=2))
        try:base.apply_vehicle_controls(client,base.stopped_controls());client.sim_world.tick();time.sleep(.1)
        except Exception:report['cleanup_error']=traceback.format_exc()
        if worker:worker.terminate();worker.join(timeout=5)
        if worker and worker.is_alive():worker.kill();worker.join(timeout=5)
        for q in queues:q.close()
        if depth is not None:depth.destroy()
        for actor in [sensor,obstacle]:
            if actor:
                try:
                    if actor==sensor:actor.stop()
                    actor.destroy()
                except RuntimeError:pass
        if states:
            report.update(closest_distance_m=min(s['goal_distance_m'] for s in states),
                final_distance_m=states[-1]['goal_distance_m'],
                path_length_m=sum(math.dist(a['pose'][:2],b['pose'][:2]) for a,b in zip(states,states[1:])))
        report['arrived']=report['outcome']=='arrived'
        report['reached_without_collision']=report['outcome']=='arrived' and not collisions
        (output/'report.json').write_text(json.dumps(report,indent=2))
    return {k:v for k,v in report.items() if k not in ('states','predictions','config','video_frames')}


def main(args):
    output=Path(args.output)
    if any((output/name).exists() for name in ('baseline','small','medium','large','goal.png','manifest.json','error.txt','initialization.json','native_fault.log')):
        output=output/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=True)
    args.output=str(output)  # trials must write into the same (possibly new run_*) folder
    print(f'Output directory: {output.resolve()}',flush=True)
    # Keep a native traceback even when a C/C++ extension crashes outside
    # Python's exception handling.
    fault_log=(output/'native_fault.log').open('w')
    faulthandler.enable(file=fault_log,all_threads=True)
    def stage(name):
        print(f'INIT: {name}',flush=True)
        (output/'initialization.json').write_text(json.dumps(dict(stage=name)))
    if args.headless or not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        os.environ['SDL_VIDEODRIVER']='dummy'
        os.environ['SDL_AUDIODRIVER']='dummy'
    client=None;original_settings=None;summaries=[]
    try:
        if args.stream_port:start_stream(args.stream_port)
        stage('connect')
        native=carla.Client('localhost',2000);native.set_timeout(args.rpc_timeout)
        world=native.get_world()
        if args.reload_world or world.get_map().name.split('/')[-1]!=args.town:
            # A previous synchronous client may have exited without ticking.
            # Release that state before reloading the map.
            settings=world.get_settings()
            settings.synchronous_mode=False
            world.apply_settings(settings)
            native.get_trafficmanager().set_synchronous_mode(False)
            print(f'Loading {args.town} (RPC timeout {args.rpc_timeout}s)...',flush=True)
            stage('load_world')
            world=native.load_world(args.town)
            world=native.get_world()
        stage('world_settings')
        original_settings=world.get_settings()
        # CarlaClient performs an initial tick during construction. Avoid its
        # one-second default physics step destabilizing the freshly spawned car.
        stage('create_client SDL='+os.environ.get('SDL_VIDEODRIVER','default'))
        client=base.CarlaClient(fixed_delta_seconds=base.CONTROL_DT,filter='vehicle.bh.crossbike') #client=base.CarlaClient(fixed_delta_seconds=base.CONTROL_DT)
        settings=client.sim_world.get_settings();settings.fixed_delta_seconds=base.CONTROL_DT
        client.sim_world.apply_settings(settings)
        stage('create_camera')
        camera=client.world.camera_manager;camera.transform_index=0
        camera.set_sensor(0,notify=False,force_respawn=True)
        start=world.get_map().get_spawn_points()[args.spawn_index]
        start_wp=world.get_map().get_waypoint(start.location)
        goals=start_wp.next(args.goal_distance);blockers=start_wp.next(args.goal_distance-args.obstacle_before_goal)
        if len(goals)!=1 or len(blockers)!=1:raise RuntimeError('Ambiguous road; choose another spawn index')
        goal=goals[0].transform;goal.location.z+=.6
        stage('capture_goal')
        reset(client,goal);goal=client.world.player.get_transform()
        goal_path=output.resolve()/'goal.png'
        if not cv2.imwrite(str(goal_path),base.camera_state(client).image.cv_image()):raise RuntimeError('Cannot save goal')
        stage('load_model')
        config=base.get_default_config('nomad.yaml');device='cuda' if torch.cuda.is_available() else 'cpu'
        model=base.get_weights(config,base.get_model(config),device,checkpoint_path=args.checkpoint);model.eval()
        (output/'manifest.json').write_text(json.dumps(dict(args=vars(args),goal_pose=pose_json(goal),
            start_pose=pose_json(start),
            test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            camera_transform_index=camera.transform_index,
            source_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
            goal_sha256=hashlib.sha256(goal_path.read_bytes()).hexdigest(),
            note='Obstacle shapes differ along with dimensions; goal pose used only for evaluation'),indent=2))
        stage('ready')
        for label,bp in [('baseline',None),('small','vehicle.harley-davidson.low_rider'),('medium','vehicle.mini.cooper_s'),('large','vehicle.mercedes.sprinter')]:
            print('START',label,flush=True)
            result=trial(client,model,config,device,start,goal,goal_path,label,bp,blockers[0],args)
            summaries.append(result);(output/'summary.json').write_text(json.dumps(summaries,indent=2))
            print('RESULT',json.dumps(result),flush=True)
            if label=='baseline' and not result['reached_without_collision']:
                print('Baseline failed: later trials are exploratory, not isolated obstacle effects.',flush=True)
    except Exception:
        (output/'error.txt').write_text(traceback.format_exc())
        print('Initialization/run failed; see '+str(output/'error.txt')+
              '. If CARLA timed out, check/restart the simulator before rerunning.',flush=True)
        raise
    finally:
        if client:
            try:
                base.apply_vehicle_controls(client,base.stopped_controls());client.world.destroy()
                if original_settings:client.sim_world.apply_settings(original_settings)
            finally:pygame.quit()


if __name__=='__main__':
    set_start_method('spawn',force=True)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--headless',action='store_true',help='Use SDL dummy display; camera recording remains enabled')
    p.add_argument('--reload-world',action='store_true',help='Reload the map before testing to clear prior simulation state')
    p.add_argument('--output',default='closed_loop_test/obstacles')
    p.add_argument('--town',default='Town03');p.add_argument('--spawn-index',type=int,default=0)
    p.add_argument('--goal-distance',type=float,default=12.)
    p.add_argument('--checkpoint',type=Path,help='Absolute path to local model weights; never download when specified')
    p.add_argument('--goal-radius',type=float,default=2.,help='Arrival tolerance in meters (evaluation only)')
    p.add_argument('--fail-distance',type=float,default=5.,help='Fail after passing the goal plane by this many meters along goal heading')
    p.add_argument('--rpc-timeout',type=float,default=120.,help='CARLA connection/map loading timeout in seconds')
    p.add_argument('--obstacle-before-goal',type=float,default=4.)
    p.add_argument('--timeout-sim',type=float,default=0.,help='Optional simulation limit; 0 disables')
    p.add_argument('--timeout-wall',type=float,default=0.,help='Optional wall-clock limit; 0 disables')
    p.add_argument('--stuck-seconds',type=float,default=10.,help='Stationary window after first prediction, in simulation seconds')
    p.add_argument('--stuck-distance',type=float,default=.3,help='Maximum position extent in meters to classify as stuck')
    p.add_argument('--path-selection',choices=['current','official'],default='current')
    p.add_argument('--stream-port',type=int,default=0,help='Serve a live MJPEG view on localhost:PORT (e.g. 8080); 0 disables')
    p.add_argument('--depth-camera',action='store_true',help='Attach a ground-truth depth camera co-located with the RGB camera and log an occupancy grid (control is unchanged)')
    p.add_argument('--save-depth',action='store_true',help='Also save per-frame depth/occupancy .npz files (implies --depth-camera)')
    p.add_argument('--grid-resolution',type=float,default=0.1,help='Occupancy grid cell size (m)')
    p.add_argument('--grid-h-min',type=float,default=0.2,help='Keep points higher than this above the road (m)')
    p.add_argument('--grid-h-max',type=float,default=2.5,help='Keep points lower than this above the road (m)')
    p.add_argument('--grid-max-range',type=float,default=25.,help='Ignore depth beyond this (m)')
    p.add_argument('--check-horizon',type=float,default=0.,help='Extend each candidate straight to this length (m) for collision checking only; 0 = use NoMaD paths as-is')
    p.add_argument('--safety-margin',type=float,default=0.3,help='Added to ego half-width when inflating obstacles (m)')
    args=p.parse_args()
    if args.save_depth:args.depth_camera=True
    if not (0<args.grid_resolution and 0<=args.grid_h_min<args.grid_h_max and 0<args.grid_max_range and args.safety_margin>=0):
        p.error('Invalid grid parameters')
    if not 0<args.obstacle_before_goal<args.goal_distance:p.error('Obstacle must lie between start and goal')
    if not math.isfinite(args.goal_radius) or args.goal_radius<=0:p.error('--goal-radius must be finite and positive')
    if args.goal_radius>=args.obstacle_before_goal:p.error('--goal-radius must be smaller than --obstacle-before-goal')
    if any(not math.isfinite(v) or v<=0 for v in (args.goal_distance,args.obstacle_before_goal,args.rpc_timeout,args.stuck_seconds,args.stuck_distance,args.fail_distance)):
        p.error('Distances and timeouts must be finite and positive')
    if any(not math.isfinite(v) or v<0 for v in (args.timeout_sim,args.timeout_wall)):
        p.error('Timeouts must be finite and nonnegative (0 disables)')
    if args.checkpoint is not None:
        if not args.checkpoint.is_absolute():p.error('--checkpoint must be an absolute path')
        if not args.checkpoint.is_file():p.error(f'Checkpoint not found: {args.checkpoint}')
        args.checkpoint=str(args.checkpoint)
    main(args)
