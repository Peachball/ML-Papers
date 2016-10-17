import theano.tensor as T
import theano
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    rtrl2()
