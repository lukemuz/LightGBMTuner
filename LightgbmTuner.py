import lightgbm
import numpy as np


class LightgbmTuner():
    def __init__(self):
        ##set default paramater constraints
        self.param_tune_list=[
            'learning_rate',
            'num_leaves',
            'min_data_in_leaf',
            'bagging_fraction',
            'feature_fraction']

        self.best_params={
            
            'learning_rate':0.1,
            'num_leaves':16,
            'min_data_in_leaf':100,
            'bagging_fraction':0.25,
            'feature_fraction':0.9,
            'bagging_freq':1
        }
    
        self.max_params={
            'learning_rate':1,
            'num_leaves':120,
            'min_data_in_leaf':10000,
            'bagging_fraction':1,
            'feature_fraction':1,
            'bagging_freq':1000
        }

        self.min_params={
            'learning_rate':.01,
            'num_leaves':2,
            'min_data_in_leaf':1,
            'bagging_fraction':.01,
            'feature_fraction':.01,
            'bagging_freq':1
        }

        self.delta_params={
            'learning_rate':.05,
            'num_leaves':2,
            'min_data_in_leaf':5,
            'bagging_fraction':.05,
            'feature_fraction':.05,
            'bagging_freq':1
        }

    def set_param_config(self,best_params,max_params,min_params,delta_params):
        pass

    def _test_parameter_change(self,best_cv,lightgbm_data,eval_metric,nfold,boost_params,param_to_tune,cv_minimize=True,increase_param=True):
        ##This function performs parameter-wise optimization.  
        # It does not perform a full grid search, 
        # but rather just proceeds in a single direction to find an optimium
        
        #current best param
        best_param=boost_params[param_to_tune]
        if increase_param:
            #increment best parameter for testing
            test_param=best_param + self.delta_params[param_to_tune]
        else:
            #increment best parameter for testing
            test_param=best_param - self.delta_params[param_to_tune]

        if (test_param<=self.max_params[param_to_tune] and increase_param) or (test_param>=self.min_params[param_to_tune] and not increase_param):

            #set up test params
            test_boost_params=boost_params
            test_boost_params[param_to_tune]=test_param

            cv_out=lightgbm.cv(test_boost_params,lightgbm_data,num_boost_round=3000,
                nfold=nfold,metrics=eval_metric,stratified=False)
            
            if cv_minimize:
                test_cv=np.min(cv_out[eval_metric+'-mean'])
                if test_cv<best_cv:
                    ##better, try to increment again
                    boost_params=test_boost_params
                    best_cv=test_cv
                    boost_params=self._test_parameter_change(best_cv,lightgbm_data,eval_metric,nfold,boost_params,param_to_tune,cv_minimize,increase_param)
                else:
                    ##not improving, return this iteration
                    return boost_params
            
            else:
                test_cv=np.max(cv_out[eval_metric+'-mean'])
                if test_cv>best_cv:
                    boost_params=test_boost_params
                    best_cv=test_cv
                    ##better, try to increment again
                    boost_params=self._test_parameter_change(best_cv,lightgbm_data,eval_metric,nfold,boost_params,param_to_tune,cv_minimize,increase_param)
                else:
                    ##not improving, return this iteration
                    return boost_params

        else:
            return boost_params
        
        return boost_params

        
    
    def tune_parameter(self,lightgbm_data,eval_metric,nfold,boost_params,param_to_tune,cv_minimize=True):
        ##This function optimizes a single parameter, holding the others constant.  
        ##It runs _test_parameter_change in both directions in order to find a local optimum
        
        ##initial fit
        cv_out=lightgbm.cv(boost_params,lightgbm_data,num_boost_round=3000,
            nfold=nfold,metrics=eval_metric,stratified=False)
       
        if cv_minimize:
            best_cv=np.min(cv_out[eval_metric+'-mean'])
        else:
            best_cv=np.max(cv_out[eval_metric+'-mean'])

        ##test increasing param

        best_params= self._test_parameter_change(best_cv,lightgbm_data,eval_metric,nfold,boost_params,param_to_tune,cv_minimize,increase_param=True)

        best_params= self._test_parameter_change(best_cv,lightgbm_data,eval_metric,nfold,best_params,param_to_tune,cv_minimize,increase_param=False)

        return best_params


    def fit(self,lightgbm_data,eval_metric,obj,nfold,device_type='cpu',
    random_seed=1234567,early_stopping_rounds=100,cv_minimize=True,tuning_rounds=2):
        
        ##set initial paramaters
        boost_params=self.best_params
        boost_settings={
            'objective':obj,
            'device_type':device_type,
            'random_seed':random_seed,
            'early_stopping_round':early_stopping_rounds
        }

        boost_params.update(boost_settings)
        for _ in range(tuning_rounds):
            for tune_param in self.param_tune_list:
                tune_out=self.tune_parameter(lightgbm_data,eval_metric,nfold,boost_params,tune_param,cv_minimize)
                boost_params.update(tune_out)
                
        return boost_params










