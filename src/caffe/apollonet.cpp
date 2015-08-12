#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/apollonet.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
ApolloNet<Dtype>::ApolloNet() {
  Init();
}

template <typename Dtype>
Dtype ApolloNet<Dtype>::ForwardLayer(const string& layer_param_string) {
    /* This function will
     * 1) Check if the layer name is in the cache
     * 2) Create the layer if it is new
     * 3) Set up the top blobs
     * 4) Set up the bottom blobs
     * 5) Set up the parameters
     * 6) Call the Forward function */ 
     
    LayerParameter active_layer_param;
    ASSERT(active_layer_param.ParseFromString(layer_param_string), "");
    RuntimeParameter runtime_param = active_layer_param.rp();
    ASSERT(active_layer_param.has_name(), "");
    const string& layer_name = active_layer_param.name();
    shared_ptr<Layer<Dtype> > layer;
    const bool new_layer = layers_map_.find(layer_name) == layers_map_.end();
    if (new_layer) {
      layer = LayerRegistry<Dtype>::CreateLayer(active_layer_param);;
      LOG(INFO) << "Creating Layer " << layer_name;
      LOG(INFO) << active_layer_param.DebugString();
      layers_map_[layer_name] = layer;
      active_layers_set_.insert(layer_name);
    } else {
      layer = layers_map_[layer_name];
      std::pair<set<string>::iterator,bool> ret = active_layers_set_.insert(layer_name);
      ASSERT(ret.second, "Layer with name '" << layer_name << "' is already used");
      ASSERT(layer->layer_param().type() == active_layer_param.type(), 
          "WARNING: layer with name '" << active_layer_param.name() << "' and different type already exists.");
    }
    layer->set_runtime_param(runtime_param);

    active_layers_vec_.push_back(layer_name);
    vector<Blob<Dtype>*> bottom_vec;
    vector<Blob<Dtype>*> top_vec;

    const vector<string>& bottom_names = bottom_blob_names_[layer_name];
    bool reset_bottoms = active_layer_param.bottom_size() != bottom_names.size();
    for (int bottom_id = 0; bottom_id < active_layer_param.bottom_size(); ++bottom_id) {
      const string& blob_name = active_layer_param.bottom(bottom_id);
      ASSERT(active_tops_set_.find(blob_name) != active_tops_set_.end(), 
          "Could not find bottom: '" << blob_name << "' for layer: " << layer_name);
      if (bottom_names.size() > bottom_id && bottom_names[bottom_id] != blob_name) { reset_bottoms = true; }
    }

    if (new_layer || reset_bottoms) {
      // layer is new, or it's list of bottoms has changed. Reset the bottom blobs
      bottom_blobs_[layer_name].clear();
      bottom_blob_names_[layer_name].clear();
      for (int i = 0; i < active_layer_param.bottom_size(); ++i) {
        const string& blob_name = active_layer_param.bottom(i);
        shared_ptr<Blob<Dtype> > top_blob = tops_[blob_name];
        bottom_blob_names_[layer_name].push_back(blob_name);
        shared_ptr<Blob<Dtype> > bottom_blob(new Blob<Dtype>(top_blob->shape()));
        bottom_blobs_[layer_name].push_back(bottom_blob);
      }
      layer->reset_bottoms(bottom_blob_names_[layer_name]);
    }

    for (int i = 0; i < active_layer_param.bottom_size(); ++i) {
      // Reshape bottom_blobs to match their respective top blobs 
      const string& blob_name = active_layer_param.bottom(i);
      shared_ptr<Blob<Dtype> > top_blob = tops_[blob_name];
      shared_ptr<Blob<Dtype> > bottom_blob = bottom_blobs_[layer_name][i];

      bottom_blob->ReshapeLike(*top_blob);
      bottom_blob->ShareData(*top_blob);
      if (layer->in_place_layer() || !layer->overwrites_bottom_diffs()) {
        // if layer accumulates delta rather than overwriting, we can save memory
        bottom_blob->ShareDiff(*top_blob);
      }
    }

    for (int bottom_id = 0; bottom_id < active_layer_param.bottom_size(); ++bottom_id) {
      bottom_vec.push_back(bottom_blobs_[layer_name][bottom_id].get());
    }

    ASSERT(layer->layer_param().top_size() == active_layer_param.top_size(), "top vec cannot change");
    for (int top_id = 0; top_id < active_layer_param.top_size(); ++top_id) {
      ASSERT(layer->layer_param().top(top_id) == active_layer_param.top(top_id), "top vec cannot change");
    }

    for (int top_id = 0; top_id < active_layer_param.top_size(); ++top_id) {
      const string& blob_name = active_layer_param.top(top_id);
      if (tops_.find(blob_name) == tops_.end()) {
        shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
        tops_[blob_name] = blob_pointer;
      }
      Blob<Dtype>* top_blob = tops_[blob_name].get();
      if (!layer->in_place_layer()) {
        std::pair<set<string>::iterator,bool> ret = active_tops_set_.insert(blob_name);
        ASSERT(ret.second, "Top with name '" << blob_name << "' is already used");
      }
      top_vec.push_back(top_blob);
      if (top_blob->DiffInitialized() && !layer->is_loss()) {
        // Zero out top_diffs, except for loss blobs, which never change
        top_blob->SetDiffValues(0.);
      }
    }

    if (new_layer) {
      layer->SetUp(bottom_vec, top_vec);
      AddLayerParams(layer);
    }

    for (int param_id = 0; param_id < layer->param_names().size(); ++param_id) {
      const string& param_name = layer->param_names()[param_id];
      active_params_set_.insert(param_name);
    }

    Dtype loss = 0;
    layer->set_phase(phase_);
    //if (new_layer && caffe_model_weights_.find(layer_name) != caffe_model_weights_.end()) { TODO
      //layer->Reshape(bottom_vec, top_vec);
      //CopyTrainedLayerNameFrom(layer_name);
    //}
    loss = layer->Forward(bottom_vec, top_vec);
    return loss;
}

template <typename Dtype>
void ApolloNet<Dtype>::AddLayerParams(shared_ptr<Layer<Dtype> > layer) {
  //hook up param names and lr_mults with Net
  vector<string> param_names;
  vector<Dtype> param_decay_mults;
  vector<Dtype> param_lr_mults;
  const LayerParameter& layer_param = layer->layer_param();
  const int param_size = layer_param.param_size();
  const string& layer_name = layer_param.name();
  if (param_size > 0) {
    // new layer has explitily named it's params
    ASSERT(param_size == layer->blobs().size(), "Layer: '" << layer_name << "' declared an incorrect number of params");
    for (int i = 0; i < layer->blobs().size(); ++i) {
      string param_name;
      if (layer_param.param(i).has_name()) {
        param_name = layer_param.param(i).name();
        ASSERT(param_name.find(".p") == string::npos, "named param '" << param_name << "' cannot contain .p");
      } else {
        stringstream ss;
        ss << layer_param.name() << ".p" << i;
        param_name = ss.str();
      }
      param_names.push_back(param_name);
      param_decay_mults.push_back(layer_param.param(i).decay_mult());
      param_lr_mults.push_back(layer_param.param(i).lr_mult());
    }
  } else {
    // provide default param names
    for (int i = 0; i < layer->blobs().size(); ++i) {
      stringstream ss;
      ss << layer_param.name() << ".p" << i;
      param_names.push_back(ss.str());
      param_decay_mults.push_back(Dtype(1.));
      param_lr_mults.push_back(Dtype(1.));
    }
  }
  layer->set_param_names(param_names);
  for (int i = 0; i < layer->blobs().size(); ++i) {
    const string& param_name = layer->param_names()[i];
    if (local_params_.find(param_name) == local_params_.end()) {
      local_params_[param_name] = layer->blobs()[i];
      params_[param_name] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>(layer->blobs()[i]->shape()));
      //if (params_.find(param_name) == params_.end()) { TODO
        //params_[param_name] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>(layer->blobs()[i]->shape()));
      //}
      params_[param_name]->ShareData(*local_params_[param_name]);
      if (!layer->overwrites_param_diffs()) {
        params_[param_name]->ShareDiff(*local_params_[param_name]);
      }
    } else {
      layer->blobs()[i]->ShareData(*local_params_[param_name]);
      layer->blobs()[i]->ShareDiff(*local_params_[param_name]);
    }
    param_decay_mults_[param_name] = param_decay_mults[i];
    param_lr_mults_[param_name] = param_lr_mults[i];
  }
}

template <typename Dtype>
void ApolloNet<Dtype>::BackwardLayer(const string& layer_name) {
  shared_ptr<Layer<Dtype> > layer = layers_map_[layer_name];
  const LayerParameter& layer_param = layer->layer_param();
  vector<Blob<Dtype>*> bottom_vec;
  vector<Blob<Dtype>*> top_vec;
  for (int top_id = 0; top_id < layer_param.top_size(); ++top_id) {
    const string& blob_name = layer_param.top(top_id);
    top_vec.push_back(tops_[blob_name].get());
  }
  vector<shared_ptr<Blob<Dtype> > > bottom_blobs = bottom_blobs_[layer_name];
  vector<bool> propagate_down;
  for (int bottom_id = 0; bottom_id < bottom_blobs.size(); ++bottom_id) {
    bottom_vec.push_back(bottom_blobs[bottom_id].get());
    propagate_down.push_back(true);
  }
  layer->Backward(top_vec, propagate_down, bottom_vec);

  if (layer->overwrites_bottom_diffs() && !layer->in_place_layer()) {
    // if layer overwrites bottom_diff
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); ++bottom_id) {
      const string& bottom_name = layer_param.bottom(bottom_id);
      // add layer's bottom diff buffer to previous layer's top diffs
      tops_[bottom_name]->AddDiffFrom(*bottom_vec[bottom_id]);
    }
  }
  if (layer->overwrites_param_diffs()) {
    for (int i = 0; i < layer->param_names().size(); ++i) {
      const string& param_name = layer->param_names()[i];
      // add param diff to master diff
      params_[param_name]->AddDiffFrom(*layer->blobs()[i]);
    }
  }
}

template <typename Dtype>
Dtype ApolloNet<Dtype>::DiffL2Norm() {
  Dtype sumsq_diff = 0.;
  for (set<string>::iterator it = active_params_set_.begin(); it != active_params_set_.end(); ++it) {
    const string& param_name = *it;
    sumsq_diff += params_[param_name]->sumsq_diff();
  }
  return std::sqrt(sumsq_diff);
}

template <typename Dtype>
void ApolloNet<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();

    if (layers_map_.find(source_layer_name) == layers_map_.end()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }

    LOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_map_[source_layer_name]->blobs();
      
    ASSERT(target_blobs.size() == source_layer.blobs_size(),
        "Incompatible number of blobs for layer " << source_layer_name);
    for (int j = 0; j < target_blobs.size(); ++j) {
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}

//template <typename Dtype> TODO
//void ApolloNet<Dtype>::CopyTrainedLayerNameFrom(const NetParameter& param, const string& target_layer_name) {
  //int num_source_layers = param.layer_size();
  //for (int i = 0; i < num_source_layers; ++i) {
    //const LayerParameter& source_layer = param.layer(i);
    //const string& source_layer_name = source_layer.name();
    //if (source_layer_name != target_layer_name) {
      //continue;
    //}

    //LOG(INFO) << "Copying source layer " << source_layer_name;
    //vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        //layers_map_[source_layer_name]->blobs();
      
    //ASSERT(target_blobs.size() == source_layer.blobs_size(),
        //"Incompatible number of blobs for layer " << source_layer_name);
    //for (int j = 0; j < target_blobs.size(); ++j) {
      //const bool kReshape = false;
      //target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    //}
  //}
//}

INSTANTIATE_CLASS(ApolloNet);

}  // namespace caffe
