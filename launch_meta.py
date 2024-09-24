import os; opj=os.path.join; opb=os.path.basename
import subprocess
import numpy as np
from copy import deepcopy


GLOBAL_CUDA_DEVICES = '0'

dir_data = 'path_to_dataset/PISR-dataset-ECCV2024'

def sample_views(num_all_images, num_used_images):
    ## index of used images
    nd_idx = np.arange(num_all_images)

    if isinstance(num_used_images, list):
        nd_idx = np.array(list).astype(np.int32)
    elif isinstance(num_used_images, int):
        if num_used_images > 0:
            n = num_all_images
            nn = num_used_images
            nd_idx = np.round(n/nn*(np.arange(nn))).astype(np.int32)
            assert len(nd_idx) == num_used_images
    num_all_images = len(nd_idx)

    return nd_idx

def copy_exp_and_update(exp, update):
    exp = deepcopy(exp)
    exp.update(update)
    return exp

exp_template = \
{
    'trial_name': 'test',
    '--gpu': '0',
    '--config': 'configs/pisr.yaml',
    # '--resume': 'exp/Car/test/ckpt/epoch=0-step=5000.ckpt',

    # 'pol.val.export_freq': '1000',
    # 'seed': '2024',
    
    'dataset.root_dir': '',
    # 'dataset.num_used_images': '40',
    # 'dataset.num_used_images': '\"'+str(list(sample_views(68, 30)))+'\"',
    # 'dataset.scale_factor': '1.0',
    # 'dataset.export_factor': '0.5',

    # 'trainer.max_steps': '20000',
    # 'model.train_num_rays': f'{256}',
    # 'model.max_train_num_rays': f'{8192}',
    # 'model.geometry.isosurface.resolution': '256',
    # 'model.geometry.isosurface.threshold': '0.001',

    # 'system.loss.lambda_curvature': '"[ [0, 0.0, 1e-4, 500], [500, 1e-4, 0.0, 1000] ]"',

    'pol.train.lambda_pol': '"[5000, 0.0, 1.0, 10000]"',
    # 'pol.train.loss_pa_ppa': 'true',
    # 'pol.train.loss_pa_normalize': 'true',

    'pol.train.size_kernel': '"[ [5000, 1, 15, 7500], [7000, 15, 1, 10000] ]"',
    # 'pol.train.step_kernel': '3',
    'pol.train.lambda_smooth': '"[ [5000, 0.0, 1.0, 7500], [7500, 1.0, 0.0, 10000] ]"',

}      

exp_rgb = \
{
    'desc': '',
    '--gpu': GLOBAL_CUDA_DEVICES,
    '--config': 'configs/pisr.yaml',
    'trial_name': 'rgb',
    'dataset.root_dir': '',
}

exp_sm = \
{
    '--gpu': GLOBAL_CUDA_DEVICES,
    '--config': 'configs/pisr.yaml',
    'trial_name': 'rgb_sm',
    'pol.train.size_kernel': '"[ [2500, 1, 15, 5000], [5000, 15, 1, 7500] ]"',
    'pol.train.lambda_smooth': '"[ [2500, 0.0, 1.0, 5000], [5000, 1.0, 0.0, 7500] ]"',
}

exp_pol = \
{
    '--gpu': GLOBAL_CUDA_DEVICES,
    '--config': 'configs/pisr.yaml',
    'trial_name': 'rgb_pol',
    'pol.train.lambda_pol': '"[2500, 0.0, 2.0, 5000]"',
}
exp_polo = copy_exp_and_update(exp_pol, {
    'trial_name': 'rgb_polo',
    'pol.train.loss_pa_ppa':'false'
})

exp_pol_sm = \
{
    # 'desc': 'test 30 views',
    '--gpu': GLOBAL_CUDA_DEVICES,
    '--config': 'configs/pisr.yaml',
    'trial_name': 'rgb_pol_sm',
    # 'dataset.num_used_images': '30',
    'pol.train.lambda_pol': '"[2500, 0.0, 2.0, 5000]"',
    'pol.train.size_kernel': '"[ [2500, 1, 15, 5000], [5000, 15, 1, 7500] ]"',
    'pol.train.lambda_smooth': '"[ [2500, 0.0, 1.0, 5000], [5000, 1.0, 0.0, 7500] ]"',
}
rgb_polo_sm = copy_exp_and_update(exp_pol_sm, {
    'trial_name': 'rgb_polo_sm',
    'pol.train.loss_pa_ppa':'false'
})

exp_sm_complex = \
{
    '--gpu': GLOBAL_CUDA_DEVICES,
    '--config': 'configs/pisr.yaml',
    'trial_name': 'rgb_sm_complex',
    'trainer.max_steps': '30000',
    'pol.train.size_kernel': '"[ [10000, 1, 9, 15000], [15000, 9, 1, 20000] ]"',
    'pol.train.lambda_smooth': '"[ [10000, 0.0, 1.0, 15000], [15000, 1.0, 0.0, 20000] ]"',
}

exp_pol_complex = \
{
    '--gpu': GLOBAL_CUDA_DEVICES,
    '--config': 'configs/pisr.yaml',
    'trial_name': 'rgb_pol_complex',
    'trainer.max_steps': '30000',
    'pol.train.lambda_pol': '"[10000, 0.0, 2.0, 15000]"',
}
exp_polo_complex = copy_exp_and_update(exp_pol_complex, {
    'trial_name': 'rgb_polo_complex',
    'pol.train.loss_pa_ppa':'false'
})

exp_pol_sm_complex = \
{
    '--gpu': GLOBAL_CUDA_DEVICES,
    '--config': 'configs/pisr.yaml',
    'trial_name': 'rgb_pol_sm_complex',
    'trainer.max_steps': '30000',
    'pol.train.lambda_pol': '"[10000, 0.0, 2.0, 15000]"',
    'pol.train.size_kernel': '"[ [10000, 1, 15, 15000], [15000, 15, 1, 20000] ]"',
    'pol.train.lambda_smooth': '"[ [10000, 0.0, 1.0, 15000], [15000, 1.0, 0.0, 20000] ]"',
}
exp_polo_sm_complex = copy_exp_and_update(exp_pol_sm_complex, {
    'trial_name': 'rgb_polo_sm_complex',
    'pol.train.loss_pa_ppa':'false'
})


if __name__ == "__main__":

    dict_marching_cube_range = {
        'Car': '"[[-0.4,-0.6,-0.3],[0.4,0.6,0.3]]"',
        'StandingRabbit': '1.0',
        'LyingRabbit': '1.0',
    }

    dict_camera_distance = {
        'Car': '1.15',
        'StandingRabbit': '2.0',
        'LyingRabbit': '2.0',
    }

    list_dict_exps = [

        # ------------ rgb ------------
        # copy_exp_and_update( exp_rgb,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'StandingRabbit'),
        #         'export.export_mc_range': dict_marching_cube_range["StandingRabbit"],
        #         'dataset.camera_distance': dict_camera_distance['StandingRabbit'],
        #     }  ),

        # copy_exp_and_update( exp_rgb,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'LyingRabbit'),
        #         'export.export_mc_range': dict_marching_cube_range["LyingRabbit"],
        #         'dataset.camera_distance': dict_camera_distance['LyingRabbit'],
        #     }  ),

        # copy_exp_and_update( exp_rgb,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'BlackLoong'),
        #     }  ),

        # copy_exp_and_update( exp_rgb,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'RedLoong'),
        #     }  ),

        # copy_exp_and_update( exp_rgb,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'Car'),
        #         'export.export_mc_range': dict_marching_cube_range["Car"],
        #         'dataset.camera_distance': dict_camera_distance['Car'],
        #     }  ),

        # copy_exp_and_update( exp_rgb,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'Figure'),
        #         # '--resume': opj('exp/Figure', exp_rgb['trial_name'], 'ckpt/epoch=0-step=20000.ckpt'),
        #     }  ),

        # copy_exp_and_update( exp_rgb,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'Car'),
        #         'model.geometry.isosurface.resolution': '128',
        #         # 'pol.val.export_freq': '250',
        #         # 'export.export_mc_range': dict_marching_cube_range["Car"],
        #         # 'dataset.camera_distance': dict_camera_distance['Car'],
        #     }  ),

        # ------------ rgb + smooth ------------
        # copy_exp_and_update( exp_sm,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'StandingRabbit'),
        #         'export.export_mc_range': dict_marching_cube_range["StandingRabbit"],
        #         'dataset.camera_distance': dict_camera_distance['StandingRabbit'],
        #         '--resume': opj('exp/StandingRabbit', exp_sm['trial_name'], 'ckpt/epoch=0-step=20000.ckpt'),
        #     }  ),

        # copy_exp_and_update( exp_sm,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'LyingRabbit'),
        #         'export.export_mc_range': dict_marching_cube_range["LyingRabbit"],
        #         'dataset.camera_distance': dict_camera_distance['LyingRabbit'],
        #         '--resume': opj('exp/LyingRabbit', exp_sm['trial_name'], 'ckpt/epoch=0-step=20000.ckpt'),
        #     }  ),

        # copy_exp_and_update( exp_sm,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'BlackLoong'),
        #         '--resume': opj('exp/BlackLoong', exp_sm['trial_name'], 'ckpt/epoch=0-step=20000.ckpt'),
        #     }  ),

        # copy_exp_and_update( exp_sm,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'RedLoong'),
        #         '--resume': opj('exp/RedLoong', exp_sm['trial_name'], 'ckpt/epoch=0-step=20000.ckpt'),
        #     }  ),

        # copy_exp_and_update( exp_sm,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'Car'),
        #         'export.export_mc_range': dict_marching_cube_range["Car"],
        #         'dataset.camera_distance': dict_camera_distance['Car'],               
        #     }  ),

        # ------------ rgb + pol + smooth ------------
        # copy_exp_and_update( exp_pol_sm,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'StandingRabbit'),
        #         'export.export_mc_range': dict_marching_cube_range["StandingRabbit"],
        #         'dataset.camera_distance': dict_camera_distance['StandingRabbit'],
        #         # '--resume': opj('exp/StandingRabbit', exp_pol_sm['trial_name'], 'ckpt/epoch=0-step=10000.ckpt')
        #     }  ),

        # copy_exp_and_update( exp_pol_sm,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'LyingRabbit'),
        #         'export.export_mc_range': dict_marching_cube_range["LyingRabbit"],
        #         'dataset.camera_distance': dict_camera_distance['LyingRabbit'],
        #     }  ),

        # copy_exp_and_update( exp_pol_sm,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'BlackLoong'),
        #     }  ),

        # copy_exp_and_update( exp_pol_sm,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'RedLoong'),
        #     }  ),

        # copy_exp_and_update( exp_pol_sm,
        #     {
        #         # 'desc': '',
        #         # 'trainer.max_steps': '30000',
        #         'dataset.root_dir': opj(dir_data, 'Car'),
        #         'export.export_mc_range': dict_marching_cube_range["Car"],
        #         'dataset.camera_distance': dict_camera_distance['Car'],
        # }  ),

        # copy_exp_and_update( exp_pol_sm,
        #     {
        #         '--gpu': '1',
        #         'dataset.root_dir': opj(dir_data, 'Car'),
        #     }  ),

        # copy_exp_and_update( exp_pol_sm,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'Figure'),
        #     }  ),

        # ------------ rgb + pol ------------
        # copy_exp_and_update( exp_pol,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'StandingRabbit'),
        #         'export.export_mc_range': dict_marching_cube_range["StandingRabbit"],
        #         'dataset.camera_distance': dict_camera_distance['StandingRabbit'],
        #     }  ),

        # copy_exp_and_update( exp_pol,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'LyingRabbit'),
        #         'export.export_mc_range': dict_marching_cube_range["LyingRabbit"],
        #         'dataset.camera_distance': dict_camera_distance['LyingRabbit'],
        #     }  ),

        # copy_exp_and_update( exp_pol,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'BlackLoong'),
        #     }  ),

        # copy_exp_and_update( exp_pol,
        #     {
        #         'dataset.root_dir': opj(dir_data, 'RedLoong'),
        #     }  ),

        copy_exp_and_update( exp_pol,
            {
                'dataset.root_dir': opj(dir_data, 'Car'),
                'export.export_mc_range': dict_marching_cube_range["Car"],
                'dataset.camera_distance': dict_camera_distance['Car'],
            }  ),

    ]

    for dict_exp in list_dict_exps:
        str_desc = getattr(dict_exp, 'desc', '')
        str_cmd = 'python launch.py --train '
        vis_cmd = str_desc+'\n\n' + 'python launch.py --train\n'
        for k, v in dict_exp.items():
            if k[:2] == '--':
                str_cmd += k + ' ' + v + ' '
                vis_cmd += k + ' ' + v + '\n'
            else:
                str_cmd += k + '=' + v + ' '
                vis_cmd += k + '=' + v + '\n'
            str_cmd += ' '

        print('\n' + '*'*80)
        print(f"Start  bash: [\n\n{str_cmd}\n\n]")
        print(f"Start   vis: [\n\n{vis_cmd}\n\n]")
        print('*'*40)
        print()

        # Log START
        dir_log = opj(
            'exp',
            opb(dict_exp['dataset.root_dir']),
            # opb(dict_exp['trial_name']),
        )
        os.makedirs(dir_log, exist_ok=True)
        logfile_name = opj(dir_log, opb(dict_exp['trial_name'])+'.txt')
        logfile = open(logfile_name, 'w')
        logfile.write(str_cmd+'\n\n')
        logfile.write(vis_cmd+'\n\n')
        logfile.close()

        # Execution START
        output_cmd = subprocess.check_output(str_cmd, universal_newlines=True, shell=True)
        output_cmd = output_cmd.split("\n")
        list_output = [ line + "\n" for line in output_cmd[:50]+["......"]+output_cmd[-50:] ]
        logfile = open(logfile_name, 'a')
        logfile.writelines(list_output)
        logfile.close()
        # Execution  END

        print()
        print('*'*40)
        print(f"Finish bash: [\n\n{str_cmd}\n\n]")
        print(f"Finish  vis: [\n\n{vis_cmd}\n\n]")
        print('*'*80 + '\n')
        # Log  END
