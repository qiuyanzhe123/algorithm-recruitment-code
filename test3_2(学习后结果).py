import torch
import matplotlib.pyplot as plt
torch.manual_seed(42)
x1=torch.linspace(-3,3,30)
x2=torch.linspace(-3,3,30)
x1,x2=torch.meshgrid(x1,x2,indexing="ij")
y=2*x1**2+1.5*x1*x2+3*torch.sin(x1)+0.5*torch.cos(2*x2)+torch.exp(-0.5*x1)+0.8*x2+2026
noise=torch.randint(-3,4,y.shape).float()
y_noise=y+noise
a1=torch.nn.Parameter(torch.tensor(1.0))
a2=torch.nn.Parameter(torch.tensor(1.0))
a3=torch.nn.Parameter(torch.tensor(1.0))
a4=torch.nn.Parameter(torch.tensor(1.0))
a5=torch.nn.Parameter(torch.tensor(-0.2))
a6=torch.nn.Parameter(torch.tensor(1.0))
b=torch.nn.Parameter(torch.tensor(2020.0))
prediction=a1*x1**2+a2*x1*x2+a3*torch.sin(x1)+a4*torch.cos(2*x2)+torch.exp(a5*x1)+a6*x2+b
loss_fn=torch.nn.MSELoss()
loss=loss_fn(prediction,y_noise)
optimizer=torch.optim.Adam([a1,a2,a3,a4,a5,a6,b],lr=0.01)
for epoch in range(3000):
    prediction=a1*x1**2+a2*x1*x2+a3*torch.sin(x1)+a4*torch.cos(2*x2)+torch.exp(a5*x1)+a6*x2+b
    loss=loss_fn(prediction,y_noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("a1 =",a1.item())
print("a2 =",a2.item())
print("a3 =",a3.item())
print("a4 =",a4.item())
print("a5 =",a5.item())
print("a6 =",a6.item())
print("b =",b.item())
with torch.no_grad():
    prediction=a1*x1**2+a2*x1*x2+a3*torch.sin(x1)+a4*torch.cos(2*x2)+torch.exp(a5*x1)+a6*x2+b
    ss_res=torch.sum((y_noise-prediction)**2)
    ss_tot=torch.sum((y_noise-torch.mean(y_noise))**2)
    r2=1-ss_res/ss_tot
print("R² =",r2.item())
with torch.no_grad():
    fitted=a1*x1**2+a2*x1*x2+a3*torch.sin(x1)+a4*torch.cos(2*x2)+torch.exp(a5*x1)+a6*x2+b
fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.plot_surface(x1.numpy(),x2.numpy(),fitted.numpy(),cmap="plasma")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.set_title(f"3.2result (R²={r2.item():.4f})")
plt.savefig("test3_2_学习结果.jpg",dpi=300,bbox_inches="tight")
plt.show()

