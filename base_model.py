
import torch.nn.functional as F
from lightning import LightningModule

class AsrBaseModule(LightningModule):
    def __init__(self,model,optimizer,criterion):
        super().__init__()
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
    def training_step(self,batch,batch_idx):
        
        mels,labels,input_lens,label_lens = batch
        
        output = self.model(mels)
        output = F.log_softmax(output,dim=2)
        output = output.transpose(0,1)
        
        loss = self.criterion(output,labels,input_lens,label_lens)
        
        return loss
        
    
    def validation_step(self,batch,batch_idx):
        
        mels,labels,input_lens,label_lens = batch
        
        output = self.model(mels)
        output = F.log_softmax(output,dim=2)
        output = output.transpose(0,1)
        
        val_loss = self.criterion(output,labels,input_lens,label_lens)
        
        return val_loss
    
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters())