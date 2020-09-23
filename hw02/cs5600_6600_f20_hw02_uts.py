
####################################################
# CS 5600/6600: F20: Assignment 2: Unit Tests
# bugs to vladimir kulyukin through canvas
#####################################################

import numpy as np
from  cs5600_6600_f20_hw02 import *
import pickle
import unittest

class cs5600_6600_f20_hw02_uts(unittest.TestCase):

    def test_assgn_02_ut_01(self):
        wmats = build_nn_wmats((2, 3, 1))
        print(wmats[0])
        assert wmats[0].shape == (2, 3)
        print(wmats[1])
        assert wmats[1].shape == (3, 1)

    def test_assgn_02_ut_02(self):
        wmats = build_nn_wmats((8, 3, 8))
        print(wmats[0])
        assert wmats[0].shape == (8, 3)
        print(wmats[1])
        assert wmats[1].shape == (3, 8)

    def test_assgn_02_ut_03(self, thresh=0.4):
        and_wmats = train_3_layer_nn(500, X1, y_and, build_231_nn)
        print('Training & Testing 3-layer 2x3x1 AND ANN Thresholded at {}'.format(thresh))
        for i in range(len(X1)):
            print('{}, {} --> {}'.format(X1[i], fit_3_layer_nn(X1[i], and_wmats), y_and[i]))
            assert (fit_3_layer_nn(X1[i], and_wmats, thresh=0.4, thresh_flag=True) == y_and[i]).all()
        #save(and_wmats, r'C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw02\anns\and_3_layer_ann.pck')
        print('\n')                    

    def test_assgn_02_ut_04(self, thresh=0.4):
        and_wmats = train_4_layer_nn(500, X1, y_and, build_2331_nn)
        print('Training & Testing 4-layer 2x3x3x1 AND ANN Thresholded at {}'.format(thresh))
        for i in range(len(X1)):
            print('{}, {} --> {}'.format(X1[i], fit_4_layer_nn(X1[i], and_wmats), y_and[i]))
            assert (fit_4_layer_nn(X1[i], and_wmats, thresh=0.4, thresh_flag=True) == y_and[i]).all()
        #save(and_wmats, r'C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw02\anns\and_4_layer_ann.pck')
        print('\n')            
        
    def test_assgn_02_ut_05(self, thresh=0.4):
        or_wmats = train_3_layer_nn(500, X1, y_or, build_231_nn)
        print('Training & Testing 2x3x1 OR ANN Thresholded at {}'.format(thresh))    
        for i in range(len(X1)):
            print('{}, {} --> {}'.format(X1[i], fit_3_layer_nn(X1[i], or_wmats), y_or[i]))
            assert (fit_3_layer_nn(X1[i], or_wmats, thresh=0.4, thresh_flag=True) == y_or[i]).all()
        #save(or_wmats, r'C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw02\anns\or_3_layer_ann.pck')
        print('\n')

    def test_assgn_02_ut_05_1(self, thresh=0.4):
        or_wmats = train_4_layer_nn(500, X1, y_or, build_2331_nn)
        print('Training & Testing 2x3x1 OR ANN Thresholded at {}'.format(thresh))
        for i in range(len(X1)):
            print('{}, {} --> {}'.format(X1[i], fit_4_layer_nn(X1[i], or_wmats), y_or[i]))
            assert (fit_4_layer_nn(X1[i], or_wmats, thresh=0.4, thresh_flag=True) == y_or[i]).all()
        # save(or_wmats, r'C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw02\anns\or_4_layer_ann.pck')
        print('\n')

    def test_assgn_02_ut_06_1(self, thresh=0.4):
        xor_wmats = train_3_layer_nn(600, X1, y_xor, build_231_nn)
        print('Training & Testing 2x3x1 XOR ANN Thresholded at {}'.format(thresh))
        for i in range(len(X1)):
            print('{}, {} --> {}'.format(X1[i], fit_3_layer_nn(X1[i], xor_wmats), y_xor[i]))
            assert (fit_3_layer_nn(X1[i], xor_wmats, thresh=0.4, thresh_flag=True) == y_xor[i]).all()
        # save(xor_wmats, r'C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw02\anns\xor_3_layer_ann.pck')
        print('\n')

    def test_assgn_02_ut_06(self, thresh=0.4):
        xor_wmats = train_4_layer_nn(600, X1, y_xor, build_2331_nn)
        print('Training & Testing 2x3x3x1 XOR ANN Thresholded at {}'.format(thresh))    
        for i in (len(X1)):
            print('{}, {} --> {}'.format(X1[i], fit_4_layer_nn(X1[i], xor_wmats), y_xor[i]))
            assert (fit_4_layer_nn(X1[i], xor_wmats, thresh=0.4, thresh_flag=True) == y_xor[i]).all()
        # save(xor_wmats, r'C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw02\anns\xor_4_layer_ann.pck')
        print('\n')

    def test_assgn_02_ut_07(self, thresh=0.4):
        not_wmats = train_3_layer_nn(500, X2, y_not, build_121_nn)
        print('Training & Testing 1x2x1 NOT ANN Thresholded at {}'.format(thresh))        
        for i in range(len(X2)):
            print('{}, {} --> {}'.format(X2[i], fit_3_layer_nn(X2[i], not_wmats), y_not[i]))
            assert (fit_3_layer_nn(X2[i], not_wmats, thresh=0.4, thresh_flag=True) == y_not[i]).all()
        #save(not_wmats, r'C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw02\anns\not_3_layer_ann.pck')
        print('\n')

    def test_assgn_02_ut_07_1(self, thresh=0.4):
        not_wmats = train_4_layer_nn(500, X2, y_not, build_1221_nn)
        print('Training & Testing 1x2x2x1 NOT ANN Thresholded at {}'.format(thresh))
        for i in range(len(X2)):
            print('{}, {} --> {}'.format(X2[i], fit_4_layer_nn(X2[i], not_wmats), y_not[i]))
            assert (fit_4_layer_nn(X2[i], not_wmats, thresh=0.4, thresh_flag=True) == y_not[i]).all()
        #save(not_wmats, r'C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw02\anns\not_4_layer_ann.pck')
        print('\n')

    def test_assgn_02_ut_09(self, thresh=0.4):
        bool_wmats = train_3_layer_nn(700, X3, bool_exp, build_421_nn)
        print('Training & Testing 4x2x1 BOOL EXP ANN Thresholded at {}'.format(thresh))        
        for i in range(len(X3)):
            print('{}, {} --> {}'.format(X3[i], fit_3_layer_nn(X3[i], bool_wmats), bool_exp[i]))
            assert (fit_3_layer_nn(X3[i], bool_wmats, thresh=0.4, thresh_flag=True) == bool_exp[i]).all()
        #save(bool_wmats, r'C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw02\anns\bool_3_layer_ann.pck')
        print('\n')

    def test_assgn_02_ut_10(self, thresh=0.4):
        bool_wmats = train_4_layer_nn(700, X3, bool_exp, build_4221_nn)
        print('Training & Testing 4x2x2x1 BOOL EXP ANN Thresholded at {}'.format(thresh))        
        for i in range(len(X3)):
            print('{}, {} --> {}'.format(X3[i], fit_4_layer_nn(X3[i], bool_wmats), bool_exp[i]))
            assert (fit_4_layer_nn(X3[i], bool_wmats, thresh=0.4, thresh_flag=True) == bool_exp[i]).all()
        #save(bool_wmats, r'C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw02\anns\bool_4_layer_ann.pck')
        print('\n')


if __name__ == '__main__':
    unittest.main()


 
    
        
        

    
        





