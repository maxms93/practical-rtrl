from jax import random, vmap, value_and_grad, jit, ops
import jax.numpy as np
from jax.experimental import optimizers
from jax.nn import sigmoid
from jax._src.util import partial

from Utils import BinaryCrossEntropyLoss, calculateSnApPattern, SparseMatrix, jacrev

"""
LSTM model with BPTT and RTRL training algorithm.
"""
class LSTM:

    def __init__(self, 
                 key,
                 input_size, 
                 output_size, 
                 hidden_size, 
                 batch_size, 
                 recurrent_density, 
                 in_out_density, 
                 snapLevel, 
                 lossFunction, 
                 training_algo="BPTT", 
                 logEvery=1, 
                 step_size=1e-3, 
                 online=False):

        self.key = key
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size= batch_size
        self.recurrent_density = recurrent_density
        self.in_out_density = in_out_density
        self.logEvery = logEvery
        self.online = online
        self.activation = sigmoid
        self.name = 'LSTM_'+ training_algo + '_'  + str(hidden_size) + '_' + str(recurrent_density) + '_' + str(snapLevel)
        self.initialize_weights()

        print('LSTM with '+ training_algo)
        print('Dense LSTM params: ', (4*hidden_size*(input_size+hidden_size) + hidden_size*output_size))
        print('Sparse LSTM params: ', len(self.paramsData.flatten()))
        print('Density: ', recurrent_density)

        if training_algo == 'SNAP':
            self.initSnAP(snapLevel)

            if (self.online):
                print('Online Updates!')
                self.update = self.updateOnline
                self.name = self.name + '_ON'
            else:
                print('Offline Updates!')
                self.update = self.updateOffline
                self.name = self.name + '_OFF'
        elif training_algo == 'BPTT':
            self.update = self.updateBPTT

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size)
        self.opt_state = self.opt_init(self.paramsData)

        self.opt_update = jit(self.opt_update)

        self.lossFunction = lossFunction

    def initSnAP(self, snapLevel):
        
        print('Init SnAP ', snapLevel)

        weightRows = np.concatenate((self.Wi.rows, self.Wo.rows, self.Wf.rows, self.Wz.rows))
        weightCols = np.concatenate((self.Wi.cols, self.Wo.cols, self.Wf.cols, self.Wz.cols))
        
        recurrentRows= np.concatenate((self.Ri.rows, self.Ro.rows, self.Rf.rows, self.Rz.rows))
        recurrentCols = np.concatenate((self.Ri.cols, self.Ro.cols, self.Rf.cols, self.Rz.cols))

        SnAP_rows, SnAP_cols  = calculateSnApPattern(snapLevel, weightRows, weightCols, recurrentRows, recurrentCols)
        
        self.J = SparseMatrix()
        self.J.jacobian(SnAP_rows, SnAP_cols, (self.hidden_size, self.Rz.end), 0)
        print('Jacobian Shape: ', self.J.shape)
        print('Jacobian params: ', self.J.len)
        print('Jacobian density: ', self.J.density)
 
    def initialize_weights(self):
        k1, k2, k3, k4, k5, k6, k7, k8, k9 = random.split(self.key, 9)

        self.Wi = SparseMatrix(k1, self.input_size, self.hidden_size, self.in_out_density, 0)
        Wi_data = self.Wi.init()
        self.Wo = SparseMatrix(k2, self.input_size, self.hidden_size, self.in_out_density, self.Wi.end)
        Wo_data = self.Wo.init()
        self.Wf = SparseMatrix(k3, self.input_size, self.hidden_size, self.in_out_density, self.Wo.end)
        Wf_data = self.Wf.init()
        self.Wz = SparseMatrix(k4, self.input_size, self.hidden_size, self.in_out_density, self.Wf.end)
        Wz_data = self.Wz.init()

        self.Ri = SparseMatrix(k5, self.hidden_size, self.hidden_size, self.recurrent_density, self.Wz.end)
        Ri_data = self.Ri.init()
        self.Ro = SparseMatrix(k6, self.hidden_size, self.hidden_size, self.recurrent_density, self.Ri.end)
        Ro_data = self.Ro.init()
        self.Rf = SparseMatrix(k7, self.hidden_size, self.hidden_size, self.recurrent_density, self.Ro.end)
        Rf_data = self.Rf.init()
        self.Rz = SparseMatrix(k8, self.hidden_size, self.hidden_size, self.recurrent_density, self.Rf.end)
        Rz_data = self.Rz.init()

        self.V = SparseMatrix(k9, self.output_size, self.hidden_size, self.in_out_density, self.Rz.end)
        V_data = self.V.init()
        
        self.paramsData = np.concatenate((Wi_data, Wo_data, Wf_data, Wz_data, Ri_data, Ro_data, Rf_data, Rz_data, V_data))

    @partial(jit, static_argnums=(0,))
    def lstm(self, params, x, h, c):

        inputGate = sigmoid(np.dot(x, self.Wi.toDense(params[self.Wi.start:self.Wi.end,])) + np.dot(h, self.Ri.toDense(params[self.Ri.start:self.Ri.end,])))
        outputGate = sigmoid(np.dot(x, self.Wo.toDense(params[self.Wo.start:self.Wo.end,])) + np.dot(h, self.Ro.toDense(params[self.Ro.start:self.Ro.end,])))
        forgetGate = sigmoid(np.dot(x, self.Wf.toDense(params[self.Wf.start:self.Wf.end,])) + np.dot(h, self.Rf.toDense(params[self.Rf.start:self.Rf.end,])))

        # Cell Input
        z = np.tanh(np.dot(x, self.Wz.toDense(params[self.Wz.start:self.Wz.end,])) + np.dot(h, self.Rz.toDense(params[self.Rz.start:self.Rz.end,])))

        # Cell State
        c = forgetGate * c + inputGate * z

        # Cell Output
        h = outputGate * np.tanh(c)

        return h, c

    @partial(jit, static_argnums=(0,))
    def forward_step(self, params, x,  h, c, Jh_data, Jc_data):

        ((grad_h_params, grad_h_h, grad_h_c), (grad_c_params, grad_c_h, grad_c_c)), (h, c) = jacrev(self.lstm, argnums=(0,2,3))(params, x, h, c)
        
        h_Jh = np.dot(grad_h_h, self.J.toDense(Jh_data))[tuple(self.J.coords)]
        h_Jc = np.dot(grad_h_c, self.J.toDense(Jc_data))[tuple(self.J.coords)]
        Jh = grad_h_params[tuple(self.J.coords)] + h_Jh + h_Jc

        c_Jh = np.dot(grad_c_h, self.J.toDense(Jh_data))[tuple(self.J.coords)]
        c_Jc = np.dot(grad_c_c, self.J.toDense(Jc_data))[tuple(self.J.coords)]
        Jc = grad_c_params[tuple(self.J.coords)] + c_Jh + c_Jc

        return h, c, Jh, Jc

    @partial(jit, static_argnums=(0,))
    def calculate_loss(self, params, h, y):
        output = self.activation(np.dot(self.V.toDense(params), h))
        loss = self.lossFunction(output, y)
        return loss

    @partial(jit, static_argnums=(0,))
    def combineGradients(self, grad_h, grad_out_params, Jh_data):
        grad_rec_params = np.dot(grad_h, self.J.toDense(Jh_data))
        return np.concatenate((grad_rec_params, grad_out_params))

    @partial(jit, static_argnums=(0,))
    def calculate_grads_step(self, params, x, y, h, c, Jh_data, Jc_data):

        h, c, Jh_data, Jc_data = self.forward_step(params[:self.Rz.end,], x, h, c, Jh_data, Jc_data)

        loss, (grad_out_params, grad_h) = value_and_grad(self.calculate_loss, argnums=(0,1))(params[self.Rz.end:,], h, y)
        gradient = self.combineGradients(grad_h, grad_out_params, Jh_data)
            
        return loss, gradient, h, c, Jh_data, Jc_data

    batch_calculate_grads_step = vmap(calculate_grads_step, in_axes=(None, None, 0, 0, 0, 0, 0, 0))

    def updateOnline(self, params, x, y):
        
        h = np.zeros((self.batch_size, self.hidden_size))
        c = np.zeros((self.batch_size, self.hidden_size))

        Jh_data = np.zeros((self.batch_size, self.J.len))
        Jc_data = np.zeros((self.batch_size, self.J.len))

        losses = []

        for t in range(x.shape[1]):
            loss, grads, h, c, Jh_data, Jc_data = self.batch_calculate_grads_step(params, x[:,t,:], y[:,t,:], h, c, Jh_data, Jc_data)
            losses.append(np.mean(loss, axis=0))
            self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)
            params = self.get_params(self.opt_state)

        return self.get_params(self.opt_state), np.mean(np.array(losses), axis=0)

    def calculate_grads(self, params, x, y):
        
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        Jh_data = np.zeros(self.J.len)
        Jc_data = np.zeros(self.J.len)

        losses = []
        gradients = np.zeros_like(self.paramsData)

        for t in range(x.shape[0]):
            loss, gradient, h, c, Jh_data, Jc_data = self.calculate_grads_step(params, x[t], y[t], h, c, Jh_data, Jc_data)
            losses.append(loss)
            gradients = gradients + gradient/x.shape[0]
            
        return np.mean(np.array(losses)), gradients

    batch_calculate_grads = vmap(calculate_grads, in_axes=(None, None, 0, 0))

    def updateOffline(self, params, x, y):
        
        losses, grads = self.batch_calculate_grads(params, x, y)
        self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)

        return self.get_params(self.opt_state), np.mean(np.array(losses), axis=0)

    @partial(jit, static_argnums=(0,))
    def forward_step_BPTT(self, paramsData, x, t, h, c, o):
        h, c = self.lstm(paramsData[:self.Rz.end,], x[t], h, c)
        output = self.activation(np.dot(self.V.toDense(paramsData[self.Rz.end:,]), h))
        o = ops.index_update(o, ops.index[t], output)
        return h, c, o

    def forward(self, params, x):

        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        o = np.zeros((x.shape[0], self.output_size))

        for t in range(x.shape[0]):
            h, c, o = self.forward_step_BPTT(params, x, t, h, c, o)
            
        return o
        
    def calculate_loss_BPTT(self, params, x, y):
        output = self.forward(params, x)
        loss = self.lossFunction(output, y)
        return loss

    def calculate_grads_BPTT(self, params, x, y):
        return value_and_grad(self.calculate_loss_BPTT)(params, x, y)

    batch_calculate_grads_BPTT = vmap(calculate_grads_BPTT, in_axes=(None, None, 0, 0))

    def updateBPTT(self, params, x, y):
        loss, grads = self.batch_calculate_grads_BPTT(params, x, y)
        self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)
        return self.get_params(self.opt_state), np.sum(loss, axis=0) / x.shape[0]

    def run(self, epochs, data):

        losses = []
        for i, k in zip(range(epochs), random.split(self.key, epochs)):

            x, y = data.getSample(k)
            self.paramsData, loss = self.update(self.paramsData, x, y)
            losses.append(np.sum(loss))
            if i%self.logEvery == 0:
                print('Loss ', "{:04d}".format(i) ,': ', np.sum(loss))

        return self.opt_state, self.paramsData
    
    def predict(self, x):
        return self.forward(self.paramsData, x)


if __name__ == "__main__":

    key = random.PRNGKey(1)
    np.set_printoptions(formatter={'float_kind':"{:.5f}".format})

    epochs = 10
    logEvery = 1

    maxSeqLength = 5
    minSeqLength = np.maximum(1, (maxSeqLength-5))
    bits = 1
    padding = 0
    lowValue = 0
    highValue = 1

    input_size = bits+2
    output_size = bits
    hidden_size = 16
    batch_size= 1
    seq_length = 2*maxSeqLength + 2
    recurrent_density = 1
    in_out_density = 1
    snapLevel = 2

    online = False
    step_size = 1e-3

    lossFunction = BinaryCrossEntropyLoss

    from CopyTaskData import CopyDataSet
    data = CopyDataSet(batch_size, maxSeqLength, minSeqLength, bits, padding, lowValue, highValue)

    ### BPTT
    training_algo = 'BPTT'

    model = LSTM(key,
                 input_size, 
                 output_size, 
                 hidden_size, 
                 batch_size, 
                 recurrent_density, 
                 in_out_density, 
                 snapLevel, 
                 lossFunction, 
                 training_algo, 
                 logEvery, 
                 step_size, 
                 online)
    state, params = model.run(epochs, data)

    ### Snap Offline
    training_algo = 'SNAP'

    model = LSTM(key,
                 input_size, 
                 output_size, 
                 hidden_size, 
                 batch_size, 
                 recurrent_density, 
                 in_out_density, 
                 snapLevel, 
                 lossFunction, 
                 training_algo, 
                 logEvery, 
                 step_size, 
                 online)
    state, params = model.run(epochs, data)

    ### Snap Online
    training_algo = 'SNAP'
    online = True

    model = LSTM(key,
                 input_size, 
                 output_size, 
                 hidden_size, 
                 batch_size, 
                 recurrent_density, 
                 in_out_density, 
                 snapLevel, 
                 lossFunction, 
                 training_algo, 
                 logEvery, 
                 step_size, 
                 online)
    state, params = model.run(epochs, data)
