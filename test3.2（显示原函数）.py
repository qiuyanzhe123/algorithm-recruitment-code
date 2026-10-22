import torch
import matplotlib.pyplot as plt
torch.manual_seed(42)
x1=torch.linspace(-3,3,30)
x2=torch.linspace(-3,3,30)
x1,x2=torch.meshgrid(x1,x2,indexing="ij")
y=2*x1**2+1.5*x1*x2+3*torch.sin(x1)+0.5*torch.cos(2*x2)+torch.exp(-0.5*x1)+0.8*x2+2026
noise=torch.randint(-3,4,y.shape).float()
y_noise=y+noise
fig=plt.figure()
qyz=fig.add_subplot(111,projection="3d")
qyz.plot_surface(x1.numpy(),x2.numpy(),y.numpy(),cmap="viridis")
qyz.set_xlabel("x1")
qyz.set_ylabel("x2")
qyz.set_zlabel("y")
qyz.set_title("3.2")
plt.savefig("test3_2.jpg",dpi=300,bbox_inches="tight")
plt.show()