import pickle
import numpy as np
class prognosis:
    def __init__(self,filename):
        self.filename=filename
        print(self.filename)
        with open(filename, 'rb') as file:
            self.p = pickle.load(file)

    def ten_years_death(self,attributes):

        attributes=np.array(attributes)
        attributes=attributes.reshape(1,-1)
        perc= self.p.predict_proba(attributes)

        return perc[0][0]
    def Diabetic_Retinopathy(self,input):
        mean = [4.091386, 4.607175, 4.499730, 4.606881]
        std = [0.141238, 0.105746, 0.106699, 0.103411]
        Data_input = []
        for i in range(4):
            Data_input.append((np.log(input[i]) - mean[i]) / std[i])

        Age_x_Systolic_BP = Data_input[0] * Data_input[1]
        Data_input.append(Age_x_Systolic_BP)
        Age_x_Diastolic_BP = Data_input[0] * Data_input[2]
        Data_input.append(Age_x_Diastolic_BP)
        Age_x_Cholesterol = Data_input[0] * Data_input[3]
        Data_input.append(Age_x_Cholesterol)
        Systolic_BP_x_Diastolic_BP = Data_input[1] * Data_input[2]
        Data_input.append(Systolic_BP_x_Diastolic_BP)
        Systolic_BP_x_Cholesterol = Data_input[1] * Data_input[3]
        Data_input.append(Systolic_BP_x_Cholesterol)
        Diastolic_BP_x_Cholesterol = Data_input[2] * Data_input[3]
        Data_input.append(Diastolic_BP_x_Cholesterol)
        print(Data_input)
        ans = self.p.predict_proba([Data_input])

        return ans[0][1]




#myclass = prognosis("/Users/nouromran/Documents/Graduation Project /prognosis/DeathIN10Years/10-year-risk-of-death")
myclass = prognosis("/Users/nouromran/Documents/Graduation Project /prognosis/DeathIN10Years/Diabetic_Retinopathy")
input = [ 85.180507 , 120.106129 ,  92.605936 , 125.065534]
#arr= [ 45.0,	68.0,	441.0,	1.0,	40.6,	10.0,	4.7	,200.0,	115.0,	1.77,	6.6,	2.0	,104.0,	355.0,	32.4,	5.5,	22.683195,	36.0,]
#print(myclass.ten_years_death(arr))


print(myclass.Diabetic_Retinopathy(input))