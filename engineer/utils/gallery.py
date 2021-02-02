import os
import numpy as np 

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    Parameters
        fname: File name to save
        points: [N, 3] array of points
        prob: [1, N] array of predictions in the range [0~1]
    Return:
        None
    '''
    prob = prob.transpose(0, 1).detach().numpy()
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b,prob], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d %.6f',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float prob\nend_header').format(
                          points.shape[0])
                      )
def save_gallery(preds,samples,names,gallery_id,epoch):
    pred = preds[0].cpu()
    sample = samples[0].transpose(0, 1).cpu()
    name = names[0]
    save_gallery_path = os.path.join(gallery_id,name.split('/')[-2],"epoch_{:03d}".format(epoch))
    os.makedirs(save_gallery_path,exist_ok=True)
    path = os.path.join(save_gallery_path,'pred.ply')

    save_samples_truncted_prob(path,sample,pred)