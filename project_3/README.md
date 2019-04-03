### Guidelines

1. we need 2 embeddings to make the animation. The animation corresponds to both the **words** in the lyrics and the **rhythm**.
   1. the word embedding: pretrained word2vec model (<https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit> for downloading)
   2. the mp3 embedding: the encoding step with the pretrained nsynth model (<https://colab.research.google.com/notebooks/magenta/nsynth/nsynth.ipynb#scrollTo=alpPhgo14VtU>)

2. steps for getting all the prerequisites before making the animation:

   1. download the word2vec model from 1.1

   2. upload desired background music for the animation to the nsynth colab

   3. go through all the steps in the colab notebook until the end of encoding section

   4. add a new cell and paste the following:

      ```python
      np.save('z.npy', z)
      ```

      1. find `z.npy` in sidebar: Files -> `../` -> `content/` -> `z.npy`, right click and download to local environment in the same directory as `animation.ipynb`

3. follow the insturction in `animation.ipynb` to generate animation

4. on the command line, type the following to merge the generated animation with corresponding background music

   ```bash
   ffmpeg -i video.mp4 -i audio.mp3 -c:v copy -c:a aac -strict experimental output.mp4
   ```

