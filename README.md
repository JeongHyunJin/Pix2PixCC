# pix2pixCC model

*pix2pixCC* is an improved deep learning model to use scientific datasets than previous models (*pix2pix* and *pix2pixHD*).
The model uses update loss functions: those of *pix2pixHD* model and correlation coefficient (CC) values between the real and generated data.

The *pix2pixCC* model consists of three major components: Generator, Discriminator, and Inspector.
The Generator and Discriminator are networks which get an update at every step with loss functions, and the Inspector is a module that guides the Generator to be well trained computing the CC values.
The Generator tries to generate realistic output from input, and the Discriminator tries to distinguish the more realistic pair between a real pair and a generated pair.
The real pair consists of real input and target data. The generated pair consists of real input data and output data from the Generator.

While the model is training, both networks compete with each other and get an update at every step with loss functions. 
Loss functions are objectives that score the quality of results by the model, and the networks automatically learn that they are appropriate for satisfying a goal, i.e., the generation of realistic data. 
They are iterated until the assigned iteration, which is a sufficient number assuring the convergence of the model.

--------------


