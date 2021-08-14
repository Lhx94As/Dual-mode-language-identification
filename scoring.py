import os

def get_trials(utt2lan, num_lang, output):
    with open(utt2lan, 'r') as f:
        lines = f.readlines()
    utt_list = [os.path.split(x.split(' ')[0])[-1].strip('.npy') for x in lines]
    lang_list = [int(x.split(' ')[1].strip()) for x in lines]
    targets = [i for i in range(num_lang)]
    with open(output, 'w') as f:
        for i in range(len(utt_list)):
            target_utt = lang_list[i]
            utt = utt_list[i]
            for target in targets:
                if target == target_utt:
                    f.write("{} {} target\n".format(utt, target))
                else:
                    f.write("{} {} nontarget\n".format(utt, target))

def get_score(utt2lan, scores, num_lang, output):
    with open(utt2lan, 'r') as f:
        lines = f.readlines()
    utt_list = [os.path.split(x.split(' ')[0])[-1].strip('.npy') for x in lines]
    lang_list = [int(x.split(' ')[-1].strip()) for x in lines]
    targets = [i for i in range(num_lang)]
    with open(output, 'w') as f:
        for i in range(len(utt_list)):
            score_utt = scores[i]
            for lang_id in targets:
                str_ = "{} {} {}\n".format(utt_list[i], lang_id, score_utt[lang_id])
                f.write(str_)


if __name__ == "__main__":
    import subprocess
    eer_txt = '/home/hexin/Desktop/hexin/datasets/eer_3s.txt'
    score_txt = '/home/hexin/Desktop/hexin/datasets/score_3s.txt'
    trial_txt = '/home/hexin/Desktop/hexin/datasets/trial_3s.txt'
    subprocess.call(f"/home/hexin/Desktop/hexin/kaldi/egs/subtools/computeEER.sh "
                    f"--write-file {eer_txt} {trial_txt} {score_txt}", shell=True)
