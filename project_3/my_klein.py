import numpy as np
import gensim.models.keyedvectors as word2vec
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.pyplot as plt
cos = np.cos
sin = np.sin
sqrt = np.sqrt
pi = np.pi

# limit = 5
# resolution = 10
# fps = 100
# lyrics = ['We', 'don\'t', 'need', 'no', 'education']

# # in seconds
# durations = [('', 2.8),
#              ('We', 0.4),
#              ('don\'t', 0.4),
#              ('need', 0.6),
#              ('no', 1),
#              ('education', 2),
#              ('', 4.3)]

# nframes = get_frames(durations, fps)

# word_embeddings = embed(lyrics) 
# # noise from wav embedding
# undulation = embed(wav_embedding 


          
def f(a, b):
    def eq(theta, phi):
        return np.sin(a * theta) + np.cos(b * phi)
    return eq

def get_frames(durations, fps):
    nframes = 0
    for _, length in durations:
        nframes += fps * length

    return nframes


def interp(r1, r2, nfr=50):
    t1 = np.linspace(r1[0], r2[0], nfr)
    t2 = np.linspace(r1[1], r2[1], nfr)
    
    result = np.zeros((nfr, 2))
    result[:, 0] = t1
    result[:, 1] = t2

    return result 


# def update_plot(i, Z, plot):
#     plot[0].remove()
#     plot[0] = ax.plot_surface(X, Y, Zs[:,:,i], cmap="magma")
#     ax.axis('off')


def surf(u, v, a, b, c, d):
    """
    http://paulbourke.net/geometry/klein/
    """
    half = (0 <= u) & (u < pi)
    r = 4*(1 - cos(u)/2)
    x = a*cos(u)*(1 + sin(u)) + r*cos(v + pi)
    x[half] = (
        (b*cos(u)*(1 + sin(u)) + r*cos(u)*cos(v))[half])
    y = c * sin(u)
    y[half] = (d*sin(u) + r*sin(u)*cos(v))[half]
    z = r * sin(v)
    return x, y, z

def fillXYZ(src, dst, duration, undulation, idx, X, Y, res, padding=50):
    start = idx
    end = start + padding
    # print(src)
    # print(dst)
    transition = interp(src, dst, nfr=padding)

    assert(transition.shape[0] == end - start)
    for ti in range(start, end):
        frame = transition[ti - start]
        noise = undulation[ti]
        a = frame[0] * 5 + 10
        b = frame[1] * 5 + 10
        c = noise[0] * 5 + 10
        d = noise[1] * 5 + 10
        res[:, :, :, ti] = surf(X, Y, 10, 10, 10, 10)

    start = end  # the end of padding
    end = start + duration - padding

    # insert word embedding with duration
    frame = dst
    # print(start)
    # print(duration)
    # print(end)
    for wi in range(start, end):
        noise = undulation[wi]
        a = frame[0] * 5 + 10
        b = frame[1] * 5 + 10
        c = noise[0] * 5 + 10
        d = noise[1] * 5 + 10
        res[:, :, :, ti] = surf(X, Y, 10, 10, 10, 10)

    assert(end - idx == duration)
    return res, end 
    

def get_XYZs(word_embeddings, durations, undulation, resolution,
        nframes, limit, fps):
    res = np.zeros((3, 40, 40, nframes)) 
    u, v = np.linspace(0, 2*pi, 40), np.linspace(0, 2*pi, 40)
    X, Y = np.meshgrid(u, v)
    padding = 50
    Z_i = 0  # frame index
    
    w_e = np.array([v[1] for v in word_embeddings.items()])
    src, dst = np.median(w_e, axis=0), np.mean(w_e, axis=0)
    end = Z_i + int(durations[0][1] * fps) - padding // 2
    # first item
    frame = dst 
    for wi in range(Z_i, end):
        noise = undulation[wi]
        a = frame[0] * 5 + 10
        b = frame[1] * 5 + 10
        c = noise[0] * 5 + 10
        d = noise[1] * 5 + 10
        res[:, :, :, wi] = surf(X, Y, 10, 10, 10, 10)
    Z_i = end 

    for i, (key, seconds) in enumerate(durations):
        if (i == 0): continue
        # update src
        src = dst
        src_key, _ = durations[i - 1]

        if src_key != '':
            src = word_embeddings[src_key]
            
        if (key == ''):
            dst = np.median(w_e, axis=0)
        else:
            dst = word_embeddings[key]

        res, Z_i = fillXYZ(src, dst, int(seconds * fps),
                undulation, Z_i, X, Y, res)
        
    # fill the last frames since fillZ reserves padding frames at the end 
    # for the next transition, yet we are done with all embeddings
    frame = dst
    for wi in range(Z_i, Z_i + padding // 2):
        noise = undulation[wi]
        a = frame[0] * 5 + 10
        b = frame[1] * 5 + 10
        c = noise[0] * 5 + 10
        d = noise[1] * 5 + 10
        res[:, :, :, wi] = surf(X, Y, 10, 10, 10, 10)

    return res
