import numpy as np
import gensim.models.keyedvectors as word2vec
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
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


def plain(x, y):
    return x + y


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


def fillZ(src, dst, duration, undulation, idx, X, Y, Zs, padding=50):
    start = idx
    end = start + padding
    # print(src)
    # print(dst)
    transition = interp(src, dst, nfr=padding)

    assert(transition.shape[0] == end - start)
    for ti in range(start, end):
        frame = transition[ti - start]
        noise = undulation[ti]
        x = frame[0] + noise[0]
        y = frame[1] + noise[1] 
        eq = f(x, y)
        Zs[:, :, ti] = eq(X, Y)

    start = end  # the end of padding
    end = start + duration - padding

    # insert word embedding with duration
    frame = dst
    # print(start)
    # print(duration)
    # print(end)
    for wi in range(start, end):
        noise = undulation[wi]

        x = frame[0] + noise[0]
        y = frame[1] + noise[1] 
        eq = f(x, y)
        Zs[:, :, wi] = eq(X, Y)

    assert(end - idx == duration)
    return Zs, end 
    

def get_Zs(word_embeddings, durations, undulation, resolution,
        nframes, limit, fps):
    Zs = np.zeros((resolution, resolution, nframes)) 
    x = np.linspace(-limit, limit, resolution)
    X, Y = np.meshgrid(x, x)
    padding = 50
    Z_i = 0  # frame index

    src, dst = np.array([0.2, 0.2]), np.array([0.2, 0.2])

    end = Z_i + int(durations[0][1] * fps) - padding // 2
    # first item
    frame = dst 
    for wi in range(Z_i, end):
        noise = undulation[wi]
        x = frame[0] + noise[0]
        y = frame[1] + noise[1] 
        eq = f(x, y)
        Zs[:, :, wi] = eq(X, Y)
    Z_i = end 

    for i, (key, seconds) in enumerate(durations):
        if (i == 0): continue
        # update src
        src = dst
        src_key, _ = durations[i - 1]

        if src_key != '':
            src = word_embeddings[src_key]
            
        if (key == ''):
            dst = np.array([0.2, 0.2]) 
        else:
            dst = word_embeddings[key]

        Zs, Z_i = fillZ(src, dst, int(seconds * fps),
                undulation, Z_i, X, Y, Zs)
        
    # fill the last frames since fillZ reserves padding frames at the end 
    # for the next transition, yet we are done with all embeddings
    frame = dst
    for wi in range(Z_i, Z_i + padding // 2):
        noise = undulation[wi]
        x = frame[0] + noise[0]
        y = frame[1] + noise[1] 
        eq = f(x, y)
        Zs[:, :, wi] = eq(X, Y)

    return Zs
