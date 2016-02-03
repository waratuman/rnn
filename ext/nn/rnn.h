#pragma once

#include <ruby.h>

void Init_nn();

// = NN:Network ================================================================

static void Init_nn_network();

static VALUE rnn_network_alloc(VALUE klass);

static VALUE rnn_network_init(VALUE obj);

static void rnn_network_free(nn_network_t* n);

static VALUE rnn_network_layers(VALUE obj);

static VALUE rnn_network_add_layer(VALUE obj, VALUE layer);

static VALUE rnn_network_activate(VALUE obj, VALUE input);

// = NN:Layer ==================================================================

static void Init_nn_layer();

// static VALUE rnn_layer_alloc(VALUE klass);

// static VALUE rnn_layer_init(VALUE obj);

static void rnn_layer_free(nn_layer_t* n);

static VALUE rnn_layer_activate(VALUE self, VALUE rb_input);

// #activation
static VALUE rnn_layer_activation(VALUE layer);

// #aggregation
static VALUE rnn_layer_aggregation(VALUE layer);

// #input_size
static VALUE rnn_layer_input_size(VALUE layer);

// #output_size
static VALUE rnn_layer_output_size(VALUE layer);

// #input_dimensions
static VALUE rnn_layer_input_dimensions(VALUE layer);

// #output_dimensions
static VALUE rnn_layer_output_dimensions(VALUE layer);

// = NN::Layer::FullyConnected =================================================

static void Init_nn_layer_fully_connected();

static VALUE rnn_layer_fully_connected_init(VALUE obj, VALUE activation, VALUE aggregation, VALUE inputCount, VALUE outputCount);

// = NN::Layer::Convolutional ==================================================

static void Init_nn_layer_convolutional();

static VALUE rnn_layer_convolutional_init(VALUE obj, VALUE activation, VALUE aggregation,
    VALUE inputDimensions, VALUE kernelCount, VALUE kernelPadding,
    VALUE kernelStride, VALUE kernelSize);

// = NN::Layer::LRN ============================================================

static void Init_nn_layer_lrn();

static VALUE rnn_layer_lrn_init(VALUE self, VALUE inputDimensions, VALUE kernelSize, VALUE k, VALUE alpha, VALUE beta);

static VALUE rnn_layer_lrn_k(VALUE self);

static VALUE rnn_layer_lrn_alpha(VALUE self);

static VALUE rnn_layer_lrn_beta(VALUE self);

// = NN::Layer::SinglyConnected ================================================

static void Init_nn_layer_singly_connected();

static VALUE rnn_layer_singly_connected_init(VALUE self, VALUE activation, VALUE inputCount);
