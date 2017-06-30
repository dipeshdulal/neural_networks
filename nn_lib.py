#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as sp

class NeuralNetwork:

    def __init__(
        self,
        ip_layer,
        op_layer,
        h_layer,
        learning_rate,
        ):
        self.ip_layer = ip_layer
        self.op_layer = op_layer
        self.h_layer = h_layer
        self.lr = learning_rate

        self.weight_ih = np.random.normal(0.0, pow(self.ip_layer, -0.5), (self.h_layer, self.ip_layer))
        self.weight_ho = np.random.normal(0.0, pow(self.h_layer, -0.5), (self.op_layer, self.h_layer))
        

    def nonlin(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return sp.expit(x)

    def query(self, ip):
        ips = np.array(ip, ndmin=2).T
        self.hidden_input = np.dot(self.weight_ih, ips)
        self.hidden_output = self.nonlin(self.hidden_input)
        self.result_input = np.dot(self.weight_ho, self.hidden_output)
        self.result_output = self.nonlin(self.result_input)

        return self.result_output

    def train(self, ip, op):
        ips = np.array(ip, ndmin=2).T
        ops = np.array(op, ndmin=2).T
        self.query(ip)
        error_output = ops - self.result_output
        hidden_errors = np.dot(self.weight_ho.T, error_output)
        hidden_output = np.array(self.hidden_output, ndmin=2).T
        self.weight_ho += self.lr * np.dot(error_output
                * self.nonlin(self.result_output, deriv=True),
                hidden_output)
        self.weight_ih += self.lr * np.dot(hidden_errors
                * self.nonlin(np.array(self.hidden_output, ndmin=2),
                deriv=True), ips.T)


       # print(self.hidden_output * (1- self.hidden_output))
       # print(np.dot(hidden_errors * self.nonlin(np.array(self.hidden_output, ndmin=2), deriv=True), ips.T))

			
