import pandas as pd
import numpy as np
import torch 
import os 
import matplotlib.pyplot as plt
def get_weighted_loss(pos_weights,neg_weights,epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Tensor): overall scalar loss summed across all classes
        """
        loss=0.0
      
        pos_loss=np.sum(np.mean(-pos_weights*y_true*np.log(y_pred+epsilon),axis=0))
        neg_loss=np.sum(np.mean(-(neg_weights)*(1-y_true)*np.log(1-y_pred+epsilon),axis=0))
        loss=pos_loss+neg_loss
        return loss
    return weighted_loss    
# transforms_=transforms.Compose([transforms.CenterCrop(500),transforms.ToTensor()])
if __name__=="__main__":

    # test the loss function
    y_true =np.array(
        [[1, 1, 1],
         [1, 1, 0],
         [0, 1, 0],
         [1, 0, 1]]
    )
    w_p = np.array([0.25, 0.25, 0.5])
    w_n = np.array([0.75, 0.75, 0.5])
    y_pred_1 =0.7*np.ones(y_true.shape)
    y_pred_2 =0.3*np.ones(y_true.shape)
    L = get_weighted_loss(w_p, w_n)

    