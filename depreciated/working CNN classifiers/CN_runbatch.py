from CNN_Classifiier import CNN_Classifier_funct

Train_sets = ['~/Pictures/AB', '~/Pictures/AC', '~/Pictures/BC']
Test_sets = ['~/Pictures/C', '~/Pictures/B', '~/Pictures/A']


CNN_Classifier_funct('../../../Pictures/AB', '../../../Pictures/C', 'AB')

CNN_Classifier_funct('../../../Pictures/AC', '../../../Pictures/B', 'AC')

CNN_Classifier_funct('../../../Pictures/BC', '../../../Pictures/A', 'BC')