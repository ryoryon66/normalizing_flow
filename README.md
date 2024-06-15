# normalizing flow with coupling layers

Visualized how the prior distribution changes into the target distribution.

The prior distribution used in this code is N(0,I) and the target distribution is eclipse-shaped.

![demo](https://github.com/ryoryon66/normalizing_flow/blob/main/saved_gifs/target_distribution.gif)
![demo](https://github.com/ryoryon66/normalizing_flow/blob/main/saved_gifs/learned_distribution.gif)
![demo](https://github.com/ryoryon66/normalizing_flow/blob/main/saved_gifs/transformation_steps.gif)



## model architecture
```
NormalizingFlow(
  (layers): ModuleList(
    (0-9): 10 x CouplingLayer(
      (s): Sequential(
        (0): Linear(in_features=2, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=128, bias=True)
        (3): ReLU()
        (4): Linear(in_features=128, out_features=1, bias=True)
        (5): Tanh()
      )
      (t): Sequential(
        (0): Linear(in_features=2, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=128, bias=True)
        (3): ReLU()
        (4): Linear(in_features=128, out_features=1, bias=True)
      )
    )
  )
)
```
