import os, sys
import yaml
from ruamel.yaml import YAML
import ast


yaml_file = './config/AFNO_inf.yaml'
yaml_file_mod = './config/AFNO_inf_mod.yaml'

# TC Name, Start date A [-1day], Start date [@ 989 hPa], Start date B [+1day], Region
hurricanes = [['DUMAZILE', ['2018-03-03 06:00:00'], ['2018-03-04 06:00:00'], ['2018-03-05 06:00:00']], # [10-30S, 40-70E]
                ['ERA5FA1', ['2018-03-18 06:00:00'], ['2018-03-19 06:00:00'], ['2018-03-20 06:00:00']], # [10-30S, 90-120E]
                ['JEBI', ['2018-08-29 12:00:00'], ['2018-08-30 12:00:00'], ['2018-08-31 12:00:00']], # [10-30N, 130-160E]
                ['MANGKHUT', ['2018-09-09 00:00:00'], ['2018-09-10 00:00:00'], ['2018-09-11 00:00:00']], # [10-30N, 110-160E]
                ['TRAMI', ['2018-09-22 06:00:00'], ['2018-09-23 06:00:00'], ['2018-09-24 06:00:00']], # [10-30N, 120-150E]
                ['KONGREY', ['2018-09-29 18:00:00'], ['2018-09-30 18:00:00'], ['2018-10-01 18:00:00']], # [10-30N, 120-150E]
                ['YUTU', ['2018-10-22 12:00:00'], ['2018-10-23 12:00:00'], ['2018-10-24 12:00:00']], # [5-25N, 110-160E]
                ['LORENZO', ['2019-09-25 00:00:00'], ['2019-09-26 00:00:00'], ['2019-09-27 00:00:00']],### # [10-30N, 50W-30W]
                ['HAGIBIS', ['2019-10-05 18:00:00'], ['2019-10-06 18:00:00'], ['2019-10-07 18:00:00']], # [10-30N, 130E-160E]
                ['AMPHAN', ['2020-05-15 12:00:00'], ['2020-05-16 12:00:00'], ['2020-05-17 12:00:00']], # [10-30N, 60-90E]
                ['MAYSAK', ['2020-08-28 12:00:00'], ['2020-08-29 12:00:00'], ['2020-08-30 12:00:00']],### # [10-30N, 120-140E]
                ['HAISHEN', ['2020-09-01 12:00:00'], ['2020-09-02 12:00:00'], ['2020-09-03 12:00:00']],### # [10-30N, 125-145E]
                ['TEDDY', ['2020-09-16 00:00:00'], ['2020-09-17 00:00:00'], ['2020-09-18 00:00:00']],### # [10-30N, 65W-45W]
                ['SURIGAE', ['2021-04-14 18:00:00'], ['2021-04-15 18:00:00'], ['2021-04-16 18:00:00']], # [10-30N, 120-140E]
                ['LARRY', ['2021-09-03 00:00:00'], ['2021-09-04 00:00:00'], ['2021-09-05 00:00:00']],### # [10-30N, 80W-30W]
                ['MINDULLE', ['2021-09-24 18:00:00'], ['2021-09-25 18:00:00'], ['2021-09-26 18:00:00']],### # [10-30N, 130-150E]
                #['ERA5FA2', ['2022-01-30 00:00:00'], ['2022-01-31 00:00:00'], ['2022-02-01 00:00:00']], # [10-30S, 30-70E]
                ['NANMADOL', ['2022-09-13 12:00:00'], ['2022-09-14 12:00:00'], ['2022-09-15 12:00:00']],### # [10-30N, 125-145E]
                ['MAWAR', ['2023-05-21 12:00:00'], ['2023-05-22 12:00:00'], ['2023-05-23 12:00:00']], # [5-25N, 120-150E]
                ['KHANUN', ['2023-07-28 12:00:00'], ['2023-07-29 12:00:00'], ['2023-07-30 12:00:00']],### #[10-30N, 120-140E]
                ['LEE', ['2023-09-07 18:00:00'], ['2023-09-08 18:00:00'], ['2023-09-09 18:00:00']]]### # [10-30N, 80W-30W]

exps = ['101', '102', '103', '104', '105',
        '201', '202', '203', '204', '205',
        '301', '302', '303', '304', '305', 
        '401', '402', '403', '404', '405', 
        '501', '502', '503', '504', '505', 
        '000']
for exp in exps:
    for hurricane in hurricanes:
        for i, date in enumerate([hurricane[1], hurricane[2], hurricane[3]]):
            
            if exp in ['101', '102', '103', '104', '105']:
                exp_name = 'FULL' # full model
            elif exp in ['201', '202', '203', '204', '205']:
                exp_name = 'RR'   # random_removed model
            elif exp in ['301', '302', '303', '304', '305']:
                exp_name = 'HR'   # hurricane_removed model
            elif exp in ['401', '402', '403', '404', '405']:
                exp_name = 'RRNA'   # Random_removed_NA model
            elif exp in ['501', '502', '503', '504', '505']:
                exp_name = 'RRWP'   # Random_removed_WP model
            elif exp == '000':
                exp_name = 'ORG'   # original model

            inference_file_tag = 'Run_' + exp[-1] + '/' + hurricane[0] + '_' + exp_name + '_d_' + str(i+1)
            
            h5name = os.path.join('results', inference_file_tag + '.h5')  
            if os.path.isfile(h5name):
                continue

            yaml = YAML()
            yaml.preserve_quotes = True
            yaml.default_flow_style = None
            with open(yaml_file) as f:
                modified = yaml.load(f)
                year = date[0].split('-')[0]
                modified['perturbations']['date_strings'] = date
                modified['perturbations']['inf_file_begin_date'] = [year + '-01-01 00:00:00']
                modified['perturbations']['inference_file_tag'] = inference_file_tag
                
                if year == '2018':
                    modified['afno_backbone']['inf_data_path'] = 'data/FCN_ERA5_data_v0/out_of_sample/2018/h5/2018'
                else:
                    modified['afno_backbone']['inf_data_path'] = 'TCs/' + year + '/'
                
            with open(yaml_file_mod, 'w') as f:
                yaml.dump(modified, f)
                
            print('Running exp {} on hurricane {} ...'.format(exp, hurricane[0]))
            
            NUM_TASKS_PER_NODE=sys.argv[1]  #==ngpus=4

            SCRIPT_PATH='inference/inference_loop_ddp.py'

            if exp == '000':
                WEIGHT = 'model_weights/FCN_weights_v0/backbone.ckpt'
            else:
                WEIGHT = 'results/era5_wind/afno_backbone_finetune/' + exp + '/training_checkpoints/best_ckpt.tar'
                
            run_script = 'python -m torch.distributed.launch --nproc_per_node={} {} --weights={}'.format(NUM_TASKS_PER_NODE, SCRIPT_PATH, WEIGHT)
            #run_script = 'python inference/inference_loop.py --weights=' + 'results/era5_wind/afno_backbone_finetune/' + exp + '/training_checkpoints/best_ckpt.tar'
            os.system(run_script)
