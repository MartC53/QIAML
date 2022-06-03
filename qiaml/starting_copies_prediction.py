from qiaml import decision_tree_trainer
import pandas as pd
import numpy as np

def prediction(testpath):
    models,xpara,ypara,all_para,keys = decision_tree_trainer.get_models()
    for ncps in models:
        print(models[ncps])
        
        testinput = decision_tree_trainer.tif_to_mat(testpath)
    df = pd.DataFrame(testinput, columns = all_para)
    test_df = df
    R2ar = []
    for ncps in models:
        R2ar.append(decision_tree_trainer.modeltest(models[ncps],test_df,xpara,ypara)[1])
    predicted_cps = keys[np.argmax(R2ar)]
    return print('The starting N copies should be',predicted_cps)