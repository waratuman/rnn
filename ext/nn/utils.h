#pragma once

#include <nn.h>
#include <ruby.h>

nn_aggregate_fn rnn_aggregate_fn(VALUE v);
nn_activation_fn rnn_activation_fn(VALUE v);

VALUE rnn_activation_sym(nn_activation_fn fn);
VALUE rnn_aggregate_sym(nn_aggregate_fn fn);
