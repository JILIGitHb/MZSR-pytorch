import dataset
import train
from config import *
import glob
import scipy.io

def main():
    if args.is_train==True:

        Trainer = train.Train(trial=args.trial, step=args.step, size=[HEIGHT,WIDTH,CHANNEL],
                              scale_list=SCALE_LIST, meta_batch_size=META_BATCH_SIZE, meta_lr=META_LR, meta_iter=META_ITER, task_batch_size=TASK_BATCH_SIZE,
                              task_lr=TASK_LR, task_iter=TASK_ITER,  checkpoint_dir=CHECKPOINT_DIR)

        Trainer()
    else:
        model_path = ''

        img_path=sorted(glob.glob(os.path.join(args.inputpath, '*.png')))
        gt_path=sorted(glob.glob(os.path.join(args.gtpath, '*.png')))

        scale=2.0

        try:
            kernel=scipy.io.loadmat(args.kernelpath)['kernel']
        except:
            kernel='cubic'

        Tester=test.Test(model_path, args.savepath, kernel, scale, conf, args.model, args.num_of_adaptation)
        P=[]
        for i in range(len(img_path)):
            img=imread(img_path[i])
            gt=imread(gt_path[i])

            _, pp =Tester(img, gt, img_path[i])

            P.append(pp)

        avg_PSNR=np.mean(P, 0)

        print('[*] Average PSNR ** Initial: %.4f, Final : %.4f' % tuple(avg_PSNR))

if __name__ == '__main__':
    main()