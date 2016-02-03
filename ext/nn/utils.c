#include "utils.h"

VALUE
rnn_activation_sym(nn_activation_fn fn)
{
    if (fn == nn_identity_fn) {
        return ID2SYM(rb_intern("identity"));
    }

    if (fn == nn_linear_fn) {
        return ID2SYM(rb_intern("linear"));
    }

    if (fn == nn_squared_fn) {
        return ID2SYM(rb_intern("squared"));
    }

    if (fn == nn_binary_step_fn) {
        return ID2SYM(rb_intern("binary_step"));
    }

    if (fn == nn_sigmoid_fn) {
        return ID2SYM(rb_intern("sigmoid"));
    }

    if (fn == nn_logistic_fn) {
        return ID2SYM(rb_intern("logistic"));
    }

    if (fn == nn_softstep_fn) {
        return ID2SYM(rb_intern("softstep"));
    }

    if (fn == nn_tanh_fn) {
        return ID2SYM(rb_intern("tanh"));
    }

    if (fn == nn_arctan_fn) {
        return ID2SYM(rb_intern("arctan"));
    }

    if (fn == nn_relu_fn) {
        return ID2SYM(rb_intern("relu"));
    }

    if (fn == nn_prelu_fn) {
        return ID2SYM(rb_intern("prelu"));
    }

    if (fn == nn_elu_fn) {
        return ID2SYM(rb_intern("elu"));
    }

    if (fn == nn_softplus_fn) {
        return ID2SYM(rb_intern("softplus"));
    }

    if (fn == nn_bent_identity_fn) {
        return ID2SYM(rb_intern("bent_identity"));
    }

    if (fn == nn_softexp_fn) {
        return ID2SYM(rb_intern("softexp"));
    }

    if (fn == nn_sin_fn) {
        return ID2SYM(rb_intern("sin"));
    }

    if (fn == nn_sinc_fn) {
        return ID2SYM(rb_intern("sinc"));
    }

    if (fn == nn_gaussian_fn) {
        return ID2SYM(rb_intern("gaussian"));
    }

    return Qnil;
}

VALUE
rnn_aggregate_sym(nn_aggregate_fn fn)
{
    if (fn == nn_sop_fn) {
        return ID2SYM(rb_intern("sum_of_products"));
    }

    if (fn == nn_euclidean_fn) {
        return ID2SYM(rb_intern("euclidean"));
    }

    if (fn == nn_sos_fn) {
        return ID2SYM(rb_intern("sum_of_squares"));
    }

    if (fn == nn_max_fn) {
        return ID2SYM(rb_intern("max"));
    }

    if (fn == nn_avg_fn) {
        return ID2SYM(rb_intern("avg"));
    }

    return Qnil;
}

nn_activation_fn
rnn_activation_fn(VALUE obj)
{
    const char* fname; 

    switch (TYPE(obj)) {
        case T_STRING:
            fname = StringValueCStr(obj);
            break;
        case T_SYMBOL:
            fname = rb_id2name(SYM2ID(obj));
            break;
        default:
            rb_raise(rb_eTypeError, "wrong argument type %"PRIsVALUE" (expected %"PRIsVALUE" or %"PRIsVALUE")", obj, rb_cString, rb_cSymbol);
    }

    if (strcmp(fname, "identity") == 0) {
        return nn_identity_fn;
    }

    if (strcmp(fname, "linear") == 0) {
        return nn_linear_fn;
    }

    if (strcmp(fname, "squared") == 0) {
        return nn_squared_fn;
    }

    if (strcmp(fname, "binary_step") == 0) {
        return nn_binary_step_fn;
    }

    if (strcmp(fname, "sigmoid") == 0) {
        return nn_sigmoid_fn;
    }

    if (strcmp(fname, "logistic") == 0) {
        return nn_logistic_fn;
    }

    if (strcmp(fname, "softstep") == 0) {
        return nn_softstep_fn;
    }

    if (strcmp(fname, "tanh") == 0) {
        return nn_tanh_fn;
    }

    if (strcmp(fname, "arctan") == 0) {
        return nn_arctan_fn;
    }

    if (strcmp(fname, "relu") == 0) {
        return nn_relu_fn;
    }

    if (strcmp(fname, "prelu") == 0) {
        return nn_prelu_fn;
    }

    if (strcmp(fname, "elu") == 0) {
        return nn_elu_fn;
    }

    if (strcmp(fname, "softplus") == 0) {
        return nn_softplus_fn;
    }

    if (strcmp(fname, "bent_identity") == 0) {
        return nn_bent_identity_fn;
    }

    if (strcmp(fname, "softexp") == 0) {
        return nn_softexp_fn;
    }

    if (strcmp(fname, "sin") == 0) {
        return nn_sin_fn;
    }

    if (strcmp(fname, "sinc") == 0) {
        return nn_sinc_fn;
    }

    if (strcmp(fname, "gaussian") == 0) {
        return nn_gaussian_fn;
    }

    return NULL;
}

nn_aggregate_fn
rnn_aggregate_fn(VALUE obj)
{
    const char* fname; 

    switch (TYPE(obj)) {
        case T_STRING:
            fname = StringValueCStr(obj);
            break;
        case T_SYMBOL:
            fname = rb_id2name(SYM2ID(obj));
            break;
        default:
            rb_raise(rb_eTypeError, "wrong argument type %"PRIsVALUE" (expected %"PRIsVALUE" or %"PRIsVALUE")", obj, rb_cString, rb_cSymbol);
    }

    if (strcmp(fname, "sop") == 0 || strcmp(fname, "sum_of_products") == 0) {
        return nn_sop_fn;
    }

    if (strcmp(fname, "euclidean") == 0) {
        return nn_euclidean_fn;
    }

    if (strcmp(fname, "sos") == 0 || strcmp(fname, "sum_of_squares") == 0) {
        return nn_sos_fn;
    }

    if (strcmp(fname, "max") == 0) {
        return nn_max_fn;
    }

    if (strcmp(fname, "avg") == 0) {
        return nn_avg_fn;
    }

    return NULL;
}
