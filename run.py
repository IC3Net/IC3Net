import os

if __name__ == "__main__":
    os.system("python main.py --env_name traffic_junction --nagents 20 --nprocesses 1 --num_epochs 300 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 18 --max_steps 20 --commnet --vision 3 --recurrent --add_rate_min 0.02 --add_rate_max 0.05 --curr_start 250 --curr_end 1250 --difficulty hard --max_steps 40 --mean_ratio 0 --transformer --comm_round 2")

