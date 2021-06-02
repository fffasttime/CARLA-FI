import glob
import csv
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import re

dict_measurements ={'exp_id': -1,
                    'rep': -1,
                    'weather': -1,
                    'start_point': -1,
                    'end_point': -1,
                    'collision_other': -1,
                    'collision_pedestrians': -1,
                    'collision_vehicles': -1,
                    'intersection_otherlane': -1,
                    'intersection_offroad': -1,
                    'pos_x': -1,
                    'pos_y': -1,
                    'steer': -1,
                    'throttle': -1,
                    'brake': -1,
                    'steer_g': -1,
                    'throttle_g': -1,
                    'brake_g': -1
                    }

def Loss(y, y_):
    return np.mean(np.abs(y-y_))

def stat_measurements(mfile):
    folder = os.path.split(mfile)[0]
    print('stat on expirement:', folder)
    has_goldrun = not('old' in folder)
    if not has_goldrun:
        return

    steer, throttle, brake = [], [], []
    steer_g, throttle_g, brake_g = [], [], []
    lines = 0
    exp_cnt = 0
    last_startpoint = 0.0
    
    MAE_result = []
    episode_align = []

    with open(mfile, 'r') as rfd:
        mr = csv.DictReader(rfd)
        for row in mr:
            steer.append(float(row['steer']))
            throttle.append(float(row['throttle']))
            brake.append(float(row['brake']))
            start_point = float(row['start_point'])
            if start_point!=last_startpoint:
                exp_cnt+=1
                last_startpoint=start_point
                episode_align.append(lines)

            if has_goldrun:
                steer_g.append(float(row['steer_g']))
                throttle_g.append(float(row['throttle_g']))
                brake_g.append(float(row['brake_g']))
                
            lines+=1

    steer=np.array(steer)
    throttle=np.array(throttle)
    brake=np.array(brake)
    
    steer_g=np.array(steer_g)
    throttle_g=np.array(throttle_g)
    brake_g=np.array(brake_g)
  
    # mae_episode
    episode_align.append(lines)
    assert(len(episode_align)==101)

    throttle = np.clip(throttle, 0, 1)
    brake = np.clip(brake, 0, 1)
    steer = np.clip(steer, -1, 1)

    for i in range(100):
        l = episode_align[i]
        r = episode_align[i+1]
        MAE_result.append(Loss(throttle[l:r], throttle_g[l:r])*0.45 + Loss(brake[l:r], brake_g[l:r])*0.05 + Loss(steer[l:r], steer_g[l:r])*0.5)

    with open('_benchmarks_results_full/mae_episode.csv','a') as f:
        print(re.match(r'.*EI_out_([0-9](?:_[0-9])?e_\d)_', folder).group(1).replace('_','-'),end=',',file=f)
        print(','.join(map(str, MAE_result)), file=f)
    
    '''

    cnt_percentage=lambda x: str(x)+' (%f%%)'%(x/lines*100)
    nan_count = np.count_nonzero((steer!=steer) | (throttle!=throttle) | (brake!=brake))

    print('lines:', lines, 'nan:', nan_count, "(%f %%)"%(nan_count/lines*100))
    invalid_steer = (steer<-1.1)|(steer>1.1)
    invalid_throttle = (throttle<-0.01)|(throttle>1.1)
    invalid_brake = (brake<-0.01)|(brake>1.1)
    invalid_count = (invalid_brake|invalid_throttle|invalid_steer).sum()
    print('too large:', cnt_percentage(invalid_count))
    print('  i_steer:',invalid_steer.sum(), 'i_throttle:', invalid_throttle.sum(), 'i_brake:', invalid_brake.sum())

    print('|throttle|< 0.01:', cnt_percentage((np.abs(throttle)<0.01).sum()))
    print(' throttle <-0.01:', cnt_percentage((throttle<-0.01).sum()))
    print(' throttle >  1.1:', cnt_percentage((throttle>1.1).sum()))
    print(' throttle < -1.1:', cnt_percentage((throttle<-1.1).sum()))
    print('|throttle|>  1.1:', cnt_percentage((np.abs(throttle)>1.1).sum()))
    print('|throttle|>  100:', cnt_percentage((np.abs(throttle)>100).sum()))
    print('|brake|<0.01:', cnt_percentage((np.abs(brake)<0.01).sum()))
    print(' brake > 1.1:', cnt_percentage((brake>1.1).sum()))
    print('|steer|<0.01:', cnt_percentage((np.abs(steer)<0.01).sum()))
    print(' steer  >1.1:', cnt_percentage((steer>1.1).sum()))
    print('|steer| >1.1:', cnt_percentage((np.abs(steer)>1.1).sum()))

    if not has_goldrun:
        return
    
    print('|throttle_g|< 0.01:', cnt_percentage((np.abs(throttle_g)<0.01).sum()))
    print(' throttle_g <-0.01:', cnt_percentage((throttle_g<-0.01).sum()))
    print(' throttle_g >  1.1:', cnt_percentage((throttle_g>1.1).sum()))
    print(' throttle_g < -1.1:', cnt_percentage((throttle_g<-1.1).sum()))
    print('|brake_g|<0.01:', cnt_percentage((np.abs(brake_g)<0.01).sum()))
    print(' brake_g > 1.1:', cnt_percentage((brake_g>1.1).sum()))
    print('|steer_g|<0.01:', cnt_percentage((np.abs(steer_g)<0.01).sum()))
    print('|steer_g| >1.1:', cnt_percentage((np.abs(steer_g)>1.1).sum()))
    print('throttle too small:', cnt_percentage(((throttle_g>0.05) & (np.abs(throttle)<0.1*throttle_g)).sum()))

    wrong_throttle=(throttle<throttle_g-0.01)|(throttle>throttle_g+0.01)
    wrong_steer=(steer<steer_g-0.01)|(steer>steer_g+0.01)
    wrong_brake=(brake<brake_g-0.01)|(brake>brake_g+0.01)
    wrong_count = (wrong_steer|wrong_throttle|wrong_brake).sum()
    print('wrong:', cnt_percentage(wrong_count))
    print('  wrong throttle:', cnt_percentage(wrong_throttle.sum()))
    print('  wrong    brake:', cnt_percentage(wrong_brake.sum()))
    print('  wrong    steer:', cnt_percentage(wrong_steer.sum()))

    throttle = np.clip(throttle, 0, 1)
    brake = np.clip(brake, 0, 1)
    steer = np.clip(steer, -1, 1)
    lossw = Loss(throttle, throttle_g)*0.45 + Loss(brake, brake_g)*0.05 + Loss(steer, steer_g)*0.5
    lossavg = (Loss(throttle, throttle_g) + Loss(brake, brake_g) + Loss(steer, steer_g))/3
    print('loss(weight):', lossw)
    print('loss  (avg) :', lossavg)
    print('  throttle loss:', Loss(throttle, throttle_g))
    print('  brake    loss:', Loss(brake, brake_g))
    print('  steer    loss:', Loss(steer, steer_g))

    with open('_benchmarks_results_full/stat.csv', 'a+') as f:
        print(re.match(r'.*EI_out_([0-9](?:_[0-9])?e_\d)_', folder).group(1).replace('_','-'), invalid_count/lines*100, wrong_count/lines*100, lossw, sep=',', file=f)

    '''  

def stat_jsons(jfile):
    folder = os.path.split(jfile)[0]
    print('stat on json:', folder)
    
    with open(jfile, 'r') as f:
        benchmark_dict = json.loads(f.read())
    
    summary=benchmark_dict
    completions=summary['episodes_completion']

    completion_task=[]
    for weather, tasks in completions.items():
        task=tasks[0]
        completion_task+=task
    
    with open('_benchmarks_results_full/complition_episode.csv','a') as f:
        print(re.match(r'.*EI_out_([0-9](?:_[0-9])?e_\d)_', folder).group(1).replace('_','-'),end=',',file=f)
        print(','.join(map(str, completion_task)),file=f)

if __name__ == "__main__":
    jfiles=glob.glob('_benchmarks_results_full/*/metrics.json')
    for jfile in jfiles:
        stat_jsons(jfile)
    mfiles=glob.glob('_benchmarks_results_full/*/measurements.csv')
    for mfile in mfiles:
        print('-------------------------')
        stat_measurements(mfile)
    
