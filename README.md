## Conda env setup

- conda create -n <env_name> python=3.7
- conda activate <env_name> (for Windows)
- pip install tensorflow-gpu (setup CUDA Toolkit & cuDNN before do this step)
- pip install keras
- pip install matplotlib pandas
- pip install gym[box2d]==0.17.\* PyOpenGL==3.1.\* PyOpenGL-accelerate==3.1.\*
- pip install gym[all] (work only on some agents)
