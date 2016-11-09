import theano.tensor as T
import theano
import numpy as np
import matplotlib.pyplot as plt

def generateAdagrad(params, error, alpha=0.01, epsilon=1e-8, verbose=False,
        clip=None):
    updates = []
    history = []

    if verbose:
        print("Calculating gradients")
    gradients = T.grad(error, params)
    if verbose:
        print("Done with gradients")
    count = 0
    for p, grad in zip(params, gradients):
        shape = p.get_value().shape

        totalG = theano.shared(value=np.zeros(shape).astype(theano.config.floatX))

        new_g = totalG + T.sqr(grad)
        updates.append((totalG, new_g))
        deltaw = grad / (T.sqrt(new_g) + epsilon) * alpha
        if isinstance(clip, tuple):
            deltaw = T.clip(deltaw, clip[0], clip[1])
        updates.append((p, p - deltaw))

        history.append(totalG)

    if verbose: print('')
    return (history, updates)

def generateAdadelta(params, error, decay=0.9, alpha=1, epsilon=1e-8):
    updates = []
    accUpdates = []
    accGrad = []

    gradients = T.grad(error, params)
    for p, grad in zip(params, gradients):
        shape = p.get_value().shape

        Eg = theano.shared(value=np.zeros(shape).astype(theano.config.floatX))
        Ex = theano.shared(value=np.zeros(shape).astype(theano.config.floatX))

        new_g = decay * Eg + (1 - decay) * T.sqr(grad)

        d_x = T.sqrt((Ex + epsilon) / (new_g + epsilon)) * grad * alpha
        new_x = decay * Ex + (1 - decay) * T.sqr(d_x)

        updates.append((p, p - d_x))
        updates.append((Ex, new_x))
        updates.append((Eg, new_g))

        accUpdates.append(Ex)
        accGrad.append(Eg)

    return ([accUpdates, accGrad], updates)

def generateAdam(params, error, alpha=0.001, decay1=0.9, decay2=0.999,
        epsilon=1e-8, verbose=False):
    """
        Generate updates for the adam type of stochastic gradient descent

            Variable interp.
    """
    updates = []
    moment = []
    vector = []

    if type(alpha) == 'float':
        alpha = theano.shared(np.array(alpha).astype(theano.config.floatX))
    time = theano.shared(np.array(1.0).astype(theano.config.floatX))
    epsilon = theano.shared(np.array(epsilon).astype(theano.config.floatX))
    updates.append((time, time+1))
    i = 0
    gradients = T.grad(error, params)
    decay1 = theano.shared(np.array(decay1).astype(theano.config.floatX))
    decay2 = theano.shared(np.array(decay2).astype(theano.config.floatX))
    for p, grad in zip(params, gradients):
        shape = p.get_value().shape
        grad = T.grad(error, p)

        m = theano.shared(value=np.zeros(shape).astype(theano.config.floatX))
        v = theano.shared(value=np.zeros(shape).astype(theano.config.floatX))

        m_t = decay1 * m + (1 - decay1) * grad
        v_t = decay2 * v + (1 - decay2) * T.sqr(grad)
        m_adj = m_t / (1.0 - T.pow(decay1, time))
        v_adj = v_t / (1.0 - T.pow(decay2, time))

        updates.append((m, m_t))
        updates.append((v, v_t))
        # updates.append((p, p - alpha * m_adj / (T.sqrt(v_adj) + epsilon)))

        moment.append(m)
        vector.append(v)
        if verbose: print("\rDone with {}/{}".format(i+1, len(params)), end="")
        i += 1

    if verbose: print("")
    return (moment + vector + [time, alpha, epsilon], updates)

def generateRmsProp(
        params, error, alpha=0.01, decay=0.9, fudge=1e-3, verbose=False):
    r = []
    v = []
    pr = []
    updates = []
    alpha = theano.shared(np.array(alpha).astype(theano.config.floatX))
    count = 0
    for p in params:
        grad = T.grad(error, p)

        shape = p.get_value().shape
        r_t = theano.shared(np.zeros(shape).astype(theano.config.floatX))
        v_t = theano.shared(np.zeros(shape).astype(theano.config.floatX))

        new_r = (1 - decay) * T.sqr(grad) + decay * r_t
        new_v = alpha / (T.sqrt(new_r) + fudge) * grad
        updates.append((r_t, new_r))
        updates.append((v_t, new_v))
        updates.append((p, p - new_v))
        r.append(r_t)
        v.append(v_t)

        count += 1
        if verbose: print("\rGradient {}/{} done".format(count, len(params)),
                end="")

    if verbose: print('')
    return (r + v, updates)

def generateVanillaUpdates(params, error, alpha=0.01, verbose=True):
    grad = []
    count = 0
    for p in params:
        grad.append(T.grad(error, p))
        count += 1
        print("\r{}/{} Gradients done".format(count, len(params)), end="")
    updates = [(p, p - g * alpha) for p, g in zip(params, grad)]
    print("")
    return updates

def generateMomentumUpdates(params, error, alpha=0.01, momentum=0.5):
    grad = []
    if type(alpha) == float:
        alpha = theano.shared(np.array(alpha).astype(theano.config.floatX))
    if type(momentum) == float:
        momentum = theano.shared(np.array(momentum)
                                    .astype(theano.config.floatX))
    for p in params:
        grad.append(T.grad(error, p))
    mparams = [theano.shared(np.zeros(p.shape.eval()).astype(theano.config.floatX)) for p in params]
    gradUpdates = [(p, p - g) for p, g in zip(params,mparams)]

    gradUpdates += [(m, momentum * m + alpha * g) for m, g in
        zip(mparams, grad)]
    return ([gradUpdates, mparams], gradUpdates)

def generateNesterovMomentumUpdates(params, error, alpha=0.01, momentum=0.9,
        decay=1e-6):
    print("WARNING: NOT FULLY IMPLEMENTED YET")
    updates = []
    momentum = []
    for p in params:
        v_t = theano.shared(np.zeros(p.get_value().shape)
                .astype(theano.config.floatX))
        grad = T.grad(error, p + momentum * v_t)

        updates.append((p, p + v_t))
        updates.append((v_t, momentum * v_t - alpha * grad))

        momentum.append(v_t)

    return (momentum, updates)

def generateRpropUpdates(params, error, init_size=1, verbose=False):
    prevw = []
    deltaw = []
    updates = []
    gradients = []
    #initalize stuff
    for p in params:
        prevw.append(theano.shared(np.zeros(p.shape.eval()).astype(config.floatX)))
        deltaw.append(theano.shared(init_size * np.ones(p.shape.eval()).
            astype(config.floatX)))

    iterations = 0
    for p, dw, pw in zip(params, deltaw, prevw):
        try:
            if verbose: print("\rGradient {} out of {}".format(iterations + 1, len(params)), end="")
            gradients.append(T.grad(error, p))
            iterations += 1
        except Exception:
            print('Unused input')
            continue
        #Array describing which values are when gradients are both positive or both negative
        simW = T.neq((T.eq((pw > 0), (gradients[-1] > 0))), (T.eq((pw < 0), (gradients[-1] <
            0))))

        #Array describing which values are when gradients are in opposite directions
        diffW = ((pw > 0) ^ (gradients[-1] > 0)) * (T.neq(pw, 0) * T.neq(gradients[-1], 0))
        updates.append((p, p - (T.sgn(gradients[-1]) * dw * (T.eq(diffW, 0)))))
        updates.append((dw, T.switch(diffW, dw *
            0.5, T.switch(simW, dw * 1.2, dw))))
        updates.append((pw, (T.sgn(gradients[-1]) * dw * (T.eq(diffW, 0)))))

    storage = prevw + deltaw
    if verbose: print("\nDone with updates")

    return (storage, updates)

def getRegularization(params):
    reg = T.sum(T.sqr(params[0]))
    for p in params[1:]:
        reg = reg + T.sum(T.sqr(p))
    return reg

def get_weight(shape, scale=0.05, name=None):
    return theano.shared(
            np.random.uniform(
                low=-scale,
                high=scale,
                size=shape).astype(theano.config.floatX),
            )


def sigm_derivative(x):
    return T.nnet.sigmoid(x) * (1 - T.nnet.sigmoid(x))


def tanh_derivative(x):
    return 1.0 - T.sqr(T.tanh(x))


def smooth(d, alpha=0.4):
    s = d[0]
    for i in range(len(d)):
        d[i] = d[i] * alpha + (1 - alpha) * s
        s = d[i]
    return d


def rtrl():
    """
        Initial approach to real time recurrent learning
        (Did not calculate jacobian, does not work)
    """
    IN_SIZE = 1
    OUT_SIZE = 1
    HIDDEN_SIZE = 10
    lr = theano.shared(np.array(0.01).astype('float32'))
    X = T.vector('input')
    Y_ = T.vector('label')
    updates = []
    gradUpdates = []

    def print_var(v):
        print(theano.function(
            [X, Y_], v, on_unused_input='ignore', mode='DebugMode')(
                np.array([1, 2]).astype('float32'),
                np.array([1, 3, 4]).astype('float32')))
        return

    w_xh = get_weight((HIDDEN_SIZE, IN_SIZE), name='x to h')
    w_hh = get_weight((HIDDEN_SIZE, HIDDEN_SIZE), name='h to h')
    w_ho = get_weight((OUT_SIZE, HIDDEN_SIZE), name='h to o')

    h_tm1 = get_weight((HIDDEN_SIZE,), name='hidden state')
    net_hidden = T.dot(w_hh, h_tm1) + T.dot(w_xh, X)
    hidden = T.tanh(net_hidden)
    updates.append((h_tm1, hidden))

    output = T.dot(w_ho, hidden)
    output.name = 'prediction'
    J = T.mean(T.sqr(output - Y_))
    gradUpdates.append((w_ho, w_ho - lr * T.grad(J, w_ho)))

    hidden_derivative = tanh_derivative(net_hidden)
    hh_grad = T.dot(w_hh.T, hidden_derivative)
    wxh_grad = T.outer(X, hidden_derivative).T
    whh_grad = T.dot(h_tm1, hidden_derivative.T)

    sxh_grad = get_weight((HIDDEN_SIZE, IN_SIZE))
    shh_grad = get_weight((HIDDEN_SIZE, HIDDEN_SIZE))

    updatedxh_grad = sxh_grad * hh_grad.dimshuffle(0, 'x') + wxh_grad
    updatedhh_grad = shh_grad * hh_grad.dimshuffle(0, 'x') + whh_grad

    gradUpdates.append((sxh_grad, updatedxh_grad))
    gradUpdates.append((shh_grad, updatedhh_grad))
    gradUpdates.append((
        w_xh,
        w_xh - lr * (T.grad(J, hidden).dimshuffle(0, 'x') * updatedxh_grad)))
    gradUpdates.append((
        w_hh,
        w_hh - lr * (T.grad(J, hidden).dimshuffle(0, 'x') * updatedhh_grad)))

    print('compiling')
    learn = theano.function(
            [X, Y_],
            J,
            updates=updates + gradUpdates,
            allow_input_downcast=True)

    predict = theano.function([X], output, allow_input_downcast=True)

    reset = theano.function(
              [],
              [],
              updates=[
                (sxh_grad, np.zeros((HIDDEN_SIZE, IN_SIZE)).astype('float32')),
                (shh_grad,
                    np.zeros((HIDDEN_SIZE, HIDDEN_SIZE)).astype('float32')),
                (h_tm1, np.zeros((HIDDEN_SIZE,)).astype('float32'))])

    predicted = []
    x = np.linspace(0, 10, 100).astype('float32').reshape(-1, 1)
    y = np.sin(x)

    reset()
    train_error = []
    for i in range(1000):
        reset()
        total = 0
        for j in range(x.shape[0]):
            e = learn(x[j], y[j])
            total += e
        avg = total / x.shape[0]
        train_error.append(avg)
        if i > 0 and train_error[-1] > train_error[-2]:
            lr.set_value(lr.get_value() / 2)
        print("\r{}".format(i), end="")
    print("")

    for i in range(x.shape[0]):
        predicted.append(predict(x[i])[0])

    plt.subplot(211)
    plt.plot(predicted)
    plt.subplot(212)
    plt.plot(y)
    plt.show()


def bptt():
    """
        Standard backpropagation through time to compare with rtrl
    """
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    HIDDEN_SIZE = 10
    lr = 0.01

    X = T.matrix()
    Y_ = T.matrix()

    w_xh = get_weight((HIDDEN_SIZE, INPUT_SIZE))
    w_hh = get_weight((HIDDEN_SIZE, HIDDEN_SIZE))
    w_ho = get_weight((OUTPUT_SIZE, HIDDEN_SIZE))

    hidden_state = get_weight((HIDDEN_SIZE,))

    b_h = get_weight((HIDDEN_SIZE,))
    b_o = get_weight((OUTPUT_SIZE,))

    print(w_hh.get_value().shape)

    def recurrence(x, h_tm1, w_hh, w_xh, w_ho, b_h, b_o):
        h_t = T.tanh(T.dot(w_hh, h_tm1) + T.dot(w_xh, x) + b_h)

        out = T.dot(w_ho, h_t) + b_o

        return [h_t, out]

    ([hidden, output], updates) = theano.scan(
            fn=recurrence,
            sequences=[X],
            outputs_info=[hidden_state, None],
            non_sequences=[w_hh, w_xh, w_ho, b_h, b_o],
            n_steps=X.shape[0],
            strict=True)

    new_hidden = hidden[-1]

    updates = list(updates.items())
    updates.append((hidden_state, new_hidden))
    J = T.mean(T.sqr(Y_ - output))

    predict = theano.function(
            [X], output, allow_input_downcast=True)
    params = [w_xh, w_hh, w_ho, b_h, b_o]

    gradUpdates = [(p, p - lr * g) for p, g in zip(params, T.grad(J, params))]
    learn = theano.function(
            [X, Y_], J, updates=(gradUpdates), allow_input_downcast=True)

    X_dat = np.linspace(0, 10, 100).astype('float64')[:,None]
    Y_dat = np.sin(X_dat)

    for i in range(10000):
        hidden_state.set_value(np.zeros((HIDDEN_SIZE,)).astype('float32'))
        print(learn(X_dat, Y_dat))


    plt.plot(Y_dat)
    plt.plot(predict(X_dat))
    plt.show()


def rtrl2():
    """
        Approach more consistent with paper
    """
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    HIDDEN_SIZE = 5
    lr = theano.shared(np.array(1e-2).astype('float32'))
    X = T.vector()
    Y_ = T.vector()

    theta = get_weight((
        HIDDEN_SIZE * INPUT_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE,))

    index = 0
    w_xh = theta[index:(HIDDEN_SIZE*INPUT_SIZE)].reshape((HIDDEN_SIZE, INPUT_SIZE))

    index += HIDDEN_SIZE * INPUT_SIZE
    w_hh = theta[index:(index+HIDDEN_SIZE*HIDDEN_SIZE)].reshape(
            (HIDDEN_SIZE, HIDDEN_SIZE))

    w_ho = get_weight((OUTPUT_SIZE, HIDDEN_SIZE))
    b_o = get_weight((OUTPUT_SIZE,))
    b_h = theta[-HIDDEN_SIZE:]

    prev_hidden = get_weight((HIDDEN_SIZE,))
    net_h = T.dot(w_xh, X) + T.dot(w_hh, prev_hidden) + b_h
    hidden = T.tanh(net_h)
    output = T.dot(w_ho, hidden)

    J = T.mean(T.sqr(output - Y_))

    p_upd = []
    p_upd.append((prev_hidden, hidden))

    grad_upd = []
    grad_upd.append((w_ho, w_ho - lr * T.grad(J, w_ho)))

    hid_grad = T.grad(J, hidden)
    hh_grad = theano.gradient.jacobian(hidden, prev_hidden)
    theta_grad = theano.gradient.jacobian(hidden, theta)

    stored_grad = get_weight((HIDDEN_SIZE, theta.get_value().shape[0]))
    grad_upd.append((stored_grad, T.dot(hh_grad, stored_grad) + theta_grad))
    grad_upd.append((theta, theta - T.clip(lr * T.dot(hid_grad, stored_grad).T,
        -1, 1)))

    predict = theano.function([X], output, updates=p_upd,
            allow_input_downcast=True)

    learn = theano.function([X, Y_], J, updates=(p_upd + grad_upd),
            allow_input_downcast=True)

    X_dat = np.linspace(0, 10, 100).astype('float32')[:,None]
    Y_dat = np.sin(X_dat)

    train_error = []
    z = np.zeros(stored_grad.get_value().shape).astype('float32')
    for i in range(1000):
        stored_grad.set_value(np.zeros(stored_grad.get_value().shape).astype('float32'))
        prev_hidden.set_value(np.zeros(prev_hidden.get_value().shape).astype('float32'))
        total = 0
        for j in range(X_dat.shape[0]):
            err = learn(X_dat[j], Y_dat[j])
            total += err
        train_error.append(total / X_dat.shape[0])
        if i > 0 and train_error[-1] > train_error[-2]:
            lr.set_value(lr.get_value() / 2)
        print("\r{}".format(i), end="")
        z = z + stored_grad.get_value()
        stored_grad.set_value(np.zeros(stored_grad.get_value().shape).astype('float32'))
        prev_hidden.set_value(np.zeros(prev_hidden.get_value().shape).astype('float32'))
    pred = []
    for i in range(X_dat.shape[0]):
        pred.append(predict(X_dat[i])[0])
    plt.plot(Y_dat)
    plt.plot(pred)
    plt.show()

    plt.plot(smooth(train_error, alpha=0.3))
    plt.yscale('log')
    plt.show()
    print("")
    np.set_printoptions(precision=3)
    print(z)
    np.savetxt('jacobian.csv', z)


def rtrl3():
    """Multilayer rtrl gru version"""
    INPUT_SIZE = 1
    LAYERS = (10,)
    OUTPUT_SIZE = 1
    lr = theano.shared(np.array(1e-3).astype(theano.config.floatX))

    X = T.vector()
    Y = T.vector()

    params = []
    stor_grad_updates = []
    grad_updates = []
    hid_updates = []
    def add_layer(x, size):
        l_data = {}
        dim_theta = 3 * size[0] * size[1] + 3 * size[1] * size[1]
        theta = get_weight((dim_theta,))

        index = 0
        W = theta[index:(index + 3 * size[0] * size[1])].reshape(
                (3 * size[1], size[0]))
        index += size[0] * size[1]

        U = theta[index:(index + 3 * size[1] * size[1])].reshape(
                (3 * size[1], size[1]))
        index += 3 * size[1] * size[1]

        bias = theta[index:(index + 3 * size[1])] # Gate, Hidden, Reset

        h_tm1 = get_weight((size[1],))

        wx = T.dot(W, x)
        hh = T.dot(U, h_tm1)

        z = T.nnet.sigmoid(wx[:size[1]] + hh[:size[1]] + bias[:size[1]])
        r = T.nnet.sigmoid(wx[size[1]:(2*size[1])]
                + hh[size[1]:(2*size[1])]
                + bias[size[1]:(2*size[1])])
        net_h = T.tanh(wx[-size[1]:] + hh[-size[1]:] + bias[-size[1]:])
        h_t = (1 - z) * h_tm1 + z * net_h

        weight_gradient = theano.shared(
                np.zeros((size[1], dim_theta)).astype(theano.config.floatX)
                )
        hidden_gradient = theano.shared(
                np.zeros((size[1], size[0])).astype(theano.config.floatX)
                )

        grad_upd = []
        hh_grad = theano.gradient.jacobian(h_t, h_tm1)
        xh_grad = theano.gradient.jacobian(h_t, theta)
        grad_upd.append(
                (weight_gradient, T.dot(hh_grad, weight_gradient) + xh_grad))
        grad_upd.append(
                (hidden_gradient, T.dot(hh_grad, hidden_gradient) +
                    theano.gradient.jacobian(h_t, x)))

        def reset():
            weight_gradient.set_value(
                    np.zeros((size[1], dim_theta)).astype(theano.config.floatX))
            hidden_gradient.set_value(
                    np.zeros((size[1], size[0])).astype(theano.config.floatX))
            h_tm1.set_value(
                    np.zeros((size[1],)).astype(theano.config.floatX))

        l_data['h_tm1'] = h_tm1
        l_data['param'] = theta
        l_data['reset'] = reset
        l_data['hh_grad'] = hidden_gradient
        l_data['param_grad'] = weight_gradient
        l_data['grad_upd'] = grad_upd
        l_data['hid_updates'] = [(h_tm1, h_t)]

        return h_t, l_data

    layers = []
    layers.append(add_layer(X, (INPUT_SIZE, LAYERS[0])))
    for i in range(len(LAYERS)):
        if i >= len(LAYERS) - 1:
            break
        l = layers[-1][0]
        layers.append(add_layer(l, (LAYERS[i], LAYERS[i+1])))
    w_ho = get_weight((OUTPUT_SIZE, LAYERS[-1]))
    b_o = get_weight((OUTPUT_SIZE))
    l = layers[-1][0]
    output = T.dot(w_ho, l) + b_o

    J = T.mean(T.sqr(output - Y))
    grad_updates += [(w_ho, w_ho - lr * T.grad(J, w_ho)),
                     (b_o, b_o - lr * T.grad(J, b_o))]
    dJdh = T.grad(J, l).T

    def reset():
        for l in layers:
            l[1]['reset']()

    def get_value(v):
        return (theano.function(
                [X, Y],
                v,
                # updates=(stor_grad_updates + hid_updates + grad_updates),
                allow_input_downcast=True,
                on_unused_input='ignore')(np.arange(INPUT_SIZE).astype(theano.config.floatX),
                    np.arange(OUTPUT_SIZE).astype(theano.config.floatX))
                )

    def print_value(v):
        print(get_value(v))

    for l in layers[::-1]:
        stor_grad_updates += l[1]['grad_upd']
        hid_updates += l[1]['hid_updates']
        grad_updates += [
                (l[1]['param'], l[1]['param']
                    - lr * T.dot(dJdh, l[1]['param_grad']).T)]
        dJdh = T.dot(dJdh, l[1]['hh_grad'])

    predict = theano.function([X], output, allow_input_downcast=True)
    learn = theano.function(
        [X, Y],
        J,
        updates=(stor_grad_updates + hid_updates + grad_updates),
        allow_input_downcast=True)


    #Train testing
    x = np.linspace(0, 10, 100).astype('float32')[:, None]
    y = np.sin(x)

    train_error = []
    plt.ion()
    for i in range(10000):
        reset()
        tot_err = 0
        for j in range(x.shape[0]):
            e  = (learn(x[j], y[j]))
            tot_err += e
        avg_err = 1.0 * tot_err / x.shape[0]
        print(avg_err)
        if i > 0 and train_error[-1] < avg_err:
            lr.set_value(lr.get_value().astype('float32') / 2)
        train_error.append(avg_err)
        if i % 100 == 0:
            plt.ion()
            plt.figure(0)
            plt.cla()
            plt.plot(train_error)
            plt.pause(0.05)

        if i % 1000 == 0:
            plt.ioff()
            out_val = []
            for j in range(x.shape[0]):
                out_val.append(predict(x[j]))
            plt.figure(1)
            plt.plot(out_val)
            plt.plot(y)
            plt.show()


if __name__ == '__main__':
    rtrl3()
