import torch
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
transform=transforms.ToTensor()
train_data=datasets.MNIST(root="./data",train=True,download=True,transform=transform)
test_data=datasets.MNIST(root="./data",train=False,download=True,transform=transform)
model=torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784,128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,10)
)
from torch.utils.data import DataLoader
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
loss_history=[]
train_acc_history=[]
test_acc_history=[]
for epoch in range(5):
    model.train()
    total_loss=0
    correct=0
    total=0
    for images,labels in train_loader:
        outputs=model(images)
        loss=loss_fn(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*labels.size(0)
        correct+=(outputs.argmax(1)==labels).sum().item()
        total+=labels.size(0)
    train_loss=total_loss/total
    train_acc=correct/total
    model.eval()
    test_correct=0
    test_total=0
    with torch.no_grad():
        for images,labels in test_loader:
            outputs=model(images)
            test_correct+=(outputs.argmax(1)==labels).sum().item()
            test_total+=labels.size(0)
    test_acc=test_correct/test_total
    loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
epochs=range(1,6)
torch.save(model.state_dict(),"mnist_model.pth")
plt.figure()
plt.plot(epochs,loss_history,marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("mnist_loss.jpg",dpi=300,bbox_inches="tight")
plt.show()
plt.figure()
plt.plot(epochs,train_acc_history,marker="o",label="Train Acc")
plt.plot(epochs,test_acc_history,marker="o",label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST Accuracy")
plt.legend()
plt.savefig("mnist_accuracy.jpg",dpi=300,bbox_inches="tight")
plt.show()
model.eval()
images,labels=next(iter(test_loader))
with torch.no_grad():
    outputs=model(images)
    predictions=outputs.argmax(1)
for j in range(3):
    fig,axes=plt.subplots(3,3,figsize=(8,8))
    for i,ax in enumerate(axes.flat):
        index=j*9+i
        ax.imshow(images[index].squeeze().numpy(),cmap="gray")
        ax.set_title(f"True:{labels[index].item()} Pred:{predictions[index].item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"mnist_predictions{j+1}.jpg",dpi=300,bbox_inches="tight")
    plt.show()