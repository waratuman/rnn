#include <nn.h>
#include <string.h>

#include "rnn.h"
#include "utils.h"

static VALUE rb_mNN;
static VALUE rb_mNN_Layer;
static VALUE rb_cNN_Network;
static VALUE rb_cNN_Layer_Convolutional;
static VALUE rb_cNN_Layer_FullyConnected;


void
Init_nn()
{
    rb_mNN = rb_define_module("NN");
    Init_nn_layer();
    Init_nn_network();
}

// = Network ===================================================================

void Init_nn_network()
{
    rb_cNN_Network = rb_define_class_under(rb_mNN, "Network", rb_cObject);

    rb_define_alloc_func(rb_cNN_Network, rnn_network_alloc);
    rb_define_method(rb_cNN_Network, "initialize", rnn_network_init, 0);
    rb_define_method(rb_cNN_Network, "add", rnn_network_add_layer, 1);
    rb_define_alias(rb_cNN_Network, "push", "add");
    rb_define_method(rb_cNN_Network, "layers", rnn_network_layers, 0);
    rb_define_method(rb_cNN_Network, "activate", rnn_network_activate, 1);
}

static void rnn_network_mark(VALUE obj)
{
    
}

static VALUE rnn_network_alloc(VALUE self)
{
    nn_network_t* n = nn_network_create(0, NULL);
    return Data_Wrap_Struct(self, rnn_network_mark, rnn_network_free, n);
}

static VALUE rnn_network_init(VALUE obj)
{
    rb_iv_set(obj, "@layers", rb_ary_new());

    return Qnil;
}

static void rnn_network_free(nn_network_t* n)
{
    nn_network_destroy(n);
}

static VALUE rnn_network_add_layer(VALUE obj, VALUE layer)
{
    nn_network_t* n;
    Data_Get_Struct(obj, nn_network_t, n);

    nn_layer_t* l;
    Data_Get_Struct(layer, nn_layer_t, l);
    rb_ary_push(rb_iv_get(obj, "@layers"), layer);

    nn_network_add_layer(n, l);

    return Qnil;
}

static VALUE rnn_network_layers(VALUE obj)
{
    return rb_iv_get(obj, "@layers");
}

static VALUE
rnn_network_activate(VALUE obj, VALUE rb_input)
{
    nn_network_t* n;
    Data_Get_Struct(obj, nn_network_t, n);

    int inputCount = nn_network_input_count(n);
    int outputCount = nn_network_output_count(n);

    float* input = calloc(inputCount, sizeof(float));
    float* output = calloc(outputCount, sizeof(float));

    for (int i = 0; i < inputCount; i++) {
        input[i] = (float)NUM2DBL(rb_ary_entry(rb_input, i));
    }

    nn_network_activate(n, input, output);

    VALUE rb_output = rb_ary_new();
    for (int i = 0; i < outputCount; i++) {
        rb_ary_push(rb_output, DBL2NUM(output[i]));
    }

    free(input);
    free(output);

    return rb_output;
}
// = Layer =====================================================================

void
Init_nn_layer()
{
    rb_mNN_Layer = rb_define_module_under(rb_mNN, "Layer");
    Init_nn_layer_fully_connected();
    Init_nn_layer_convolutional();
}

static void
rnn_layer_mark(nn_layer_t* n)
{
    
}

static VALUE
rnn_layer_alloc(VALUE klass)
{
    nn_layer_t* l = calloc(1, sizeof(nn_layer_t));
    return Data_Wrap_Struct(klass, rnn_layer_mark, rnn_layer_free, l);
}

static void
rnn_layer_free(nn_layer_t* l)
{
    nn_layer_destroy(l);
}

static VALUE
rnn_layer_activation(VALUE obj)
{
    nn_layer_t* l;
    Data_Get_Struct(obj, nn_layer_t, l);

    if (l->type == NN_FC) {
        return rnn_activation_sym(((nn_layer_fully_connected_t*)l->_layer)->activation);
    }

    if (l->type == NN_CV) {
        return rnn_activation_sym(((nn_layer_convolutional_t*)l->_layer)->activation);
    }

    return Qnil;
}

static VALUE
rnn_layer_aggregation(VALUE obj)
{
    nn_layer_t* l;
    Data_Get_Struct(obj, nn_layer_t, l);

    if (l->type == NN_FC) {
        return rnn_aggregate_sym(((nn_layer_fully_connected_t*)l->_layer)->aggregation);
    }

    if (l->type == NN_CV) {
        return rnn_aggregate_sym(((nn_layer_convolutional_t*)l->_layer)->aggregation);
    }

    return Qnil;
}

static VALUE
rnn_layer_input_size(VALUE obj)
{
    nn_layer_t* l;
    Data_Get_Struct(obj, nn_layer_t, l);

    return INT2NUM(nn_layer_input_count(l));
}

// #output_size
static VALUE
rnn_layer_output_size(VALUE obj)
{
    nn_layer_t* l;
    Data_Get_Struct(obj, nn_layer_t, l);

    return INT2NUM(nn_layer_output_count(l));
}

// #input_dimesions
static VALUE
rnn_layer_input_dimensions(VALUE obj)
{
    nn_layer_t* l;
    Data_Get_Struct(obj, nn_layer_t, l);

    VALUE rdims = rb_ary_new();
    int* cdims = nn_layer_input_dimensions(l);

    for (int i = 0; i < nn_layer_input_dimension_count(l); i++) {
        rb_ary_push(rdims, INT2NUM(cdims[i]));
    }

    free(cdims);

    return rdims;
}

// #output_dimesions
static VALUE
rnn_layer_output_dimensions(VALUE obj)
{
    nn_layer_t* l;
    Data_Get_Struct(obj, nn_layer_t, l);

    VALUE rdims = rb_ary_new();
    int* cdims = nn_layer_output_dimensions(l);

    for (int i = 0; i < nn_layer_output_dimension_count(l); i++) {
        rb_ary_push(rdims, INT2NUM(cdims[i]));
    }

    free(cdims);

    return rdims;
}

// = Fully Connected Layer =====================================================

void
Init_nn_layer_fully_connected()
{
    rb_cNN_Layer_FullyConnected = rb_define_class_under(rb_mNN_Layer, "FullyConnected", rb_cObject);
    rb_define_alloc_func(rb_cNN_Layer_FullyConnected, rnn_layer_alloc);
    rb_define_method(rb_cNN_Layer_FullyConnected, "initialize", rnn_layer_fully_connected_init, 4);
    rb_define_method(rb_cNN_Layer_FullyConnected, "activation", rnn_layer_activation, 0);
    rb_define_method(rb_cNN_Layer_FullyConnected, "aggregation", rnn_layer_aggregation, 0);
    rb_define_method(rb_cNN_Layer_FullyConnected, "input_size", rnn_layer_input_size, 0);
    rb_define_method(rb_cNN_Layer_FullyConnected, "output_size", rnn_layer_output_size, 0);
    rb_define_method(rb_cNN_Layer_FullyConnected, "input_dimesions", rnn_layer_input_dimensions, 0);
    rb_define_method(rb_cNN_Layer_FullyConnected, "output_dimesions", rnn_layer_output_dimensions, 0);
}

// activation: a symbol or proc / block to execute as the activation function
// aggregation: a symbol or proc / block to execute as the aggregation function
static VALUE
rnn_layer_fully_connected_init(VALUE obj, VALUE activation, VALUE aggregation, VALUE inputCount, VALUE outputCount)
{
    nn_activation_fn af = rnn_activation_fn(activation);
    nn_aggregate_fn cf = rnn_aggregate_fn(aggregation);
    int ic = NUM2UINT(inputCount);
    int oc = NUM2UINT(outputCount);

    nn_layer_t* l;
    Data_Get_Struct(obj, nn_layer_t, l);
    l->type = NN_FC;
    l->_layer = nn_layer_create_fully_connected(af, cf, ic, oc);

    return Qnil;
}

// = Convolutional Layer =======================================================

void
Init_nn_layer_convolutional()
{
    rb_cNN_Layer_Convolutional = rb_define_class_under(rb_mNN_Layer, "Convolutional", rb_cObject);
    rb_define_alloc_func(rb_cNN_Layer_Convolutional, rnn_layer_alloc);
    rb_define_method(rb_cNN_Layer_Convolutional, "initialize", rnn_layer_convolutional_init, 7);
    rb_define_method(rb_cNN_Layer_Convolutional, "activation", rnn_layer_activation, 0);
    rb_define_method(rb_cNN_Layer_Convolutional, "aggregation", rnn_layer_aggregation, 0);
    rb_define_method(rb_cNN_Layer_Convolutional, "input_size", rnn_layer_input_size, 0);
    rb_define_method(rb_cNN_Layer_Convolutional, "output_size", rnn_layer_output_size, 0);
    rb_define_method(rb_cNN_Layer_Convolutional, "input_dimesions", rnn_layer_input_dimensions, 0);
    rb_define_method(rb_cNN_Layer_Convolutional, "output_dimesions", rnn_layer_output_dimensions, 0);
}

static VALUE
rnn_layer_convolutional_init(VALUE obj, VALUE activation, VALUE aggregation,
    VALUE inputDimensions, VALUE kernelCount, VALUE kernelPadding,
    VALUE kernelStride, VALUE kernelSize)
{
    nn_activation_fn af = rnn_activation_fn(activation);
    nn_aggregate_fn cf = rnn_aggregate_fn(aggregation);

    int  ic = 1;
    int  idc = RARRAY_LEN(inputDimensions);
    int* id = calloc(idc, sizeof(int));
    int kc = NUM2UINT(kernelCount);
    int* kp = calloc(idc, sizeof(int));
    int* kt = calloc(idc, sizeof(int));
    int* ks = calloc(idc, sizeof(int));

    for (int i = 0; i < idc; i++) {
        id[i] = NUM2UINT(rb_ary_entry(inputDimensions, i));
        kp[i] = NUM2UINT(rb_ary_entry(kernelPadding, i));
        kt[i] = NUM2UINT(rb_ary_entry(kernelStride, i));
        ks[i] = NUM2UINT(rb_ary_entry(kernelSize, i));
        ic *= id[i];
    }

    nn_layer_t* l;
    Data_Get_Struct(obj, nn_layer_t, l);
    l->type = NN_CV;
    l->_layer = nn_layer_create_convolutional(af, cf, ic, idc, kc, id, kp, kt, ks);

    free(ks);
    free(kt);
    free(kp);
    free(id);

    return Qnil;
}
