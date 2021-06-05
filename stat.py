import glob
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import re

folder_name = '_benchmarks_results'

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
    exp_folder = os.path.split(mfile)[0].split('\\')[1]
    print('stat on expirement:', exp_folder)

    has_goldrun = not('old' in exp_folder)
    if not has_goldrun: return

    with open(mfile, 'r') as rfd:
        df = pd.read_csv(rfd, header = 0, delimiter=',')
    
    steer=np.array(df['steer'])
    throttle=np.array(df['throttle'])
    brake=np.array(df['brake'])
    steer_g=np.array(df['steer_g'])
    throttle_g=np.array(df['throttle_g'])
    brake_g=np.array(df['brake_g'])

    lines = len(df)
    exp_group = df.groupby('weather')
    exp_cnt = len(exp_group)

    print('data_cnt: %d, exp_cnt:%d'%(lines, exp_cnt))
    
    '''
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

    print('|throttle_g|< 0.01:', cnt_percentage((np.abs(throttle_g)<0.01).sum()))
    print(' throttle_g <-0.01:', cnt_percentage((throttle_g<-0.01).sum()))
    print(' throttle_g >  1.1:', cnt_percentage((throttle_g>1.1).sum()))
    print(' throttle_g < -1.1:', cnt_percentage((throttle_g<-1.1).sum()))
    print('|brake_g|<0.01:', cnt_percentage((np.abs(brake_g)<0.01).sum()))
    print(' brake_g > 1.1:', cnt_percentage((brake_g>1.1).sum()))
    print('|steer_g|<0.01:', cnt_percentage((np.abs(steer_g)<0.01).sum()))
    print('|steer_g| >1.1:', cnt_percentage((np.abs(steer_g)>1.1).sum()))
    print('throttle too small:', cnt_percentage(((throttle_g>0.05) & (np.abs(throttle)<0.1*throttle_g)).sum()))


    def stat_wrong(wrong_threshold, prt=0):
        wrong_throttle=abs(throttle-throttle_g)>wrong_threshold
        wrong_steer=abs(steer-steer_g)>wrong_threshold
        wrong_brake=abs(brake-brake_g)>wrong_threshold
        
        if prt:
            print('  wrong throttle:', cnt_percentage(wrong_throttle.sum()))
            print('  wrong    brake:', cnt_percentage(wrong_brake.sum()))
            print('  wrong    steer:', cnt_percentage(wrong_steer.sum()))

        return (wrong_steer|wrong_throttle|wrong_brake).sum()
   
    print('wrong(abs>0.01):', cnt_percentage(stat_wrong(0.01, 1)))
    print('wrong(abs>0.05):', cnt_percentage(stat_wrong(0.05)))

    wrong_count = stat_wrong(0.1)
    print('wrong(abs>0.1):', cnt_percentage(wrong_count))
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

    with open(folder_name + '/stat.csv', 'a+') as f:
        print(re.match(r'.*EI_out_([0-9](?:_[0-9])?e_\d)_', exp_folder).group(1).replace('_','-'), invalid_count/lines*100, wrong_count/lines*100, lossw, sep=',', file=f)

def stat_jsons(jfile):
    """
    stat completion_episode, for error bar
    """
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
    
    with open(folder_name + '/complition_episode.csv','a') as f:
        print(re.match(r'.*EI_out_([0-9](?:_[0-9])?e_\d)_', folder).group(1).replace('_','-'),end=',',file=f)
        print(','.join(map(str, completion_task)),file=f)

if __name__ == "__main__":
    print("in folder:", folder_name)

    '''
    jfiles=glob.glob(folder_name + '/*/metrics.json')
    for jfile in jfiles:
        stat_jsons(jfile)
    '''

    mfiles=glob.glob(folder_name + '/*/measurements.csv')
    for mfile in mfiles:
        print('-------------------------')
        stat_measurements(mfile)
    
