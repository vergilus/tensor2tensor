
import tensorflow as tf
from tensorflow.python.util import nest

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from transformer import Transformer
from transformer import features_to_nonpadding
from transformer import transformer_prepare_decoder
from transformer import transformer_prepare_encoder


@registry.register_model
class DualTransformer(Transformer):

  @property
  def has_input(self):
    if self._problem_hparams:
      for key in self._problem_hparams.input_modality:
        if "inputs" in key:
          return True
    else:
      return True

  def dual_encode(self,
                  wav_inputs, txt_inputs, target_space,
                  hparams,
                  features=None,
                  losses=None):
    """Encode transformer inputs.

    Args:
      wav_inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      txt_inputs: same as former
      target_space: scalar, target space ID.
      hparams: hyperparameters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.
      losses: optional list onto which to append extra training losses

    Returns:
      Tuple of:
          *_encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          *_encoder_decoder_attention_bias: Bias and mask weights for
              encoder-decoder attention. [batch_size, input_length]
    """
    wav_inputs = common_layers.flatten4d3d(wav_inputs)
    txt_inputs = common_layers.flatten4d3d(txt_inputs)

    wav_encoder_input, wav_self_attention_bias, wav_encoder_decoder_attention_bias = (
      transformer_prepare_encoder(
        wav_inputs, target_space, hparams, features=features))
    wav_encoder_input = tf.nn.dropout(wav_encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    txt_encoder_input, txt_self_attention_bias, txt_encoder_decoder_attention_bias = (
      transformer_prepare_encoder(
        txt_inputs, target_space, hparams, features=features, reuse_trg=True))
    txt_encoder_input = tf.nn.dropout(txt_encoder_input,
                                   1.0 - hparams.layer_prepostprocess_dropout)


    customized_wav_params=tf.contrib.training.HParams(
      num_heads=hparams.get("wav_num_heads"),
      num_layers=hparams.get("num_wav_enc_layers"),
      ffn_layer="conv_relu_conv",
      max_length=hparams.get("max_length") #or hparams.max_wav_seq_length*80 # for the wav_enc, we have mel-bins
    )

    wav_encoder_output = transformer_n_encoder(
      wav_encoder_input,
      wav_self_attention_bias,
      hparams,
      customize_params=customized_wav_params,
      name="wav_encoder",
      nonpadding=features_to_nonpadding(features, "wav_inputs"),
      save_weights_to=self.attention_weights,
      losses=losses)

    customized_txt_params=tf.contrib.training.HParams(
      num_heads=hparams.get("txt_num_heads"),
      num_layers=hparams.get("num_txt_enc_layers"),
      ffn_layer="dense_relu_dense",
      max_length=hparams.get("max_txt_seq_length") #or  hparams.get("max_txt_seq_length")
    )
    txt_encoder_output = transformer_n_encoder(
      txt_encoder_input,
      txt_self_attention_bias,
      hparams,
      customize_params=customized_txt_params,
      name="txt_encoder",
      nonpadding=features_to_nonpadding(features, "txt_inputs"),
      save_weights_to=self.attention_weights
    )

    return wav_encoder_output, wav_encoder_decoder_attention_bias, \
           txt_encoder_output, txt_encoder_decoder_attention_bias

  def dual_decode(self,
                  decoder_input,
                  wav_encoder_output,
                  txt_encoder_output,
                  wav_enc_dec_attention_bias,
                  txt_enc_dec_attention_bias,
                  decoder_self_attention_bias,
                  hparams,
                  cache=None,
                  nonpadding=None,
                  losses=None):
    """ dual transformer decoder, attention to both inputs """
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    decoder_output = transformer_dual_decoder(
            decoder_input,
            wav_encoder_output,txt_encoder_output,
            decoder_self_attention_bias,
            wav_enc_dec_attention_bias,
            txt_enc_dec_attention_bias,
            hparams,
            cache=cache,
            nonpadding=nonpadding,
            save_weights_to=self.attention_weights,
            losses=losses)

    if (common_layers.is_on_tpu() and
        hparams.mode == tf.estimator.ModeKeys.TRAIN):
      # TPU does not react kindly to extra dimensions.
      # TODO(noam): remove this once TPU is more forgiving of extra dims.
      return decoder_output
    else:
      # Expand since t2t expects 4d tensors.
      return tf.expand_dims(decoder_output, axis=2)


  def body(self, features):
    """dual_Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs1\2": Transformer inputs [batch_size, input_length, hidden_dim]
          "targets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    losses = []
    assert self.has_input, "problems for dual-transformer must have inputs"
    if self.has_input:
      inputs1 = features["wav_inputs"]
      inputs2 = features["txt_inputs"]
      target_space = features["target_space_id"]
      wav_encoder_output, wav_enc_dec_attention_bias,\
      txt_encoder_output, txt_enc_dec_attention_bias = self.dual_encode(
          inputs1, inputs2,target_space, hparams, features=features, losses=losses)
    else:
      wav_encoder_output, wav_enc_dec_attention_bias, \
      txt_encoder_output, txt_enc_dec_attention_bias=(None, None, None, None)

    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    targets = common_layers.flatten4d3d(targets)

    decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
        targets, hparams, features=features)

    decoder_output = self.dual_decode(
        decoder_input,
        wav_encoder_output,txt_encoder_output,
        wav_enc_dec_attention_bias,
        txt_enc_dec_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "targets"),
        losses=losses)

    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}

    ret = tf.reshape(decoder_output, targets_shape)
    if losses:
      return ret, {"extra_loss": tf.add_n(losses)}
    else:
      return ret

########## fast decode needs to combine 2 inputs #########
  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0):
    """ Fast decoding
    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.
    :param features: a map of string to model  features.
    :param decode_length:
    :param beam_size: beam search size
    :param top_beams: an integer, how many of the beams to return
    :param alpha:
    :return:
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.target_modality

    assert self.has_input, "problems for dual-transformer must have inputs"
    # decode with an input source(needs encoder outputs)
    wav_inputs = features["wav_inputs"]
    txt_inputs = features["txt_inputs"]
    if target_modality.is_class_modality:
      decode_length = 1
    else:
      decode_length = (common_layers.shape_list(wav_inputs)[1] +
                       features.get("decode_length", decode_length))

    wav_inputs = tf.expand_dims(wav_inputs, axis=1)
    txt_inputs = tf.expand_dims(txt_inputs, axis=1)
    if len(wav_inputs.shape) < 5:
      wav_inputs = tf.expand_dims(wav_inputs, axis=4)
    if len(txt_inputs.shape) < 5:
      txt_inputs = tf.expand_dims(txt_inputs, axis=4)

    s = common_layers.shape_list(wav_inputs)
    batch_size = s[0]
    wav_inputs = tf.reshape(wav_inputs, [s[0] * s[1], s[2], s[3], s[4]])
    txt_inputs = tf.reshape(txt_inputs, [s[0] * s[1], s[2], s[3], s[4]])
    # _shard_features called to ensure that the variable names match
    wav_inputs = self._shard_features({"wav_inputs": wav_inputs})["wav_inputs"]
    wav_input_modality = self._problem_hparams.input_modality["wav_inputs"]
    txt_inputs = self._shard_features({"txt_inputs": txt_inputs})["txt_inputs"]
    txt_input_modality = self._problem_hparams.input_modality["txt_inputs"]

    with tf.variable_scope(wav_input_modality.name):
      wav_inputs = wav_input_modality.bottom_sharded(wav_inputs, dp)
    with tf.variable_scope(txt_input_modality.name):
      txt_inputs = txt_input_modality.bottom_sharded(txt_inputs, dp)

    with tf.variable_scope("body"):
      wav_enc_output, wav_enc_dec_attention_bias, \
      txt_enc_output,txt_enc_dec_attention_bias = dp(
        self.dual_encode, wav_inputs, txt_inputs, features["target_space_id"], hparams,
        features=features)
    wav_enc_output = wav_enc_output[0]
    txt_enc_output = txt_enc_output[0]
    wav_enc_dec_attention_bias = wav_enc_dec_attention_bias[0]
    txt_enc_dec_attention_bias = txt_enc_dec_attention_bias[0]

    if hparams.pos == "timing":
      timing_signal = common_attention.get_timing_signal_1d(
        decode_length + 1, hparams.hidden_size)

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      with tf.variable_scope(target_modality.name):
        targets = target_modality.targets_bottom_sharded(targets, dp)[0]
      targets = common_layers.flatten4d3d(targets)

      # TODO(llion): Explain! Is this even needed?
      targets = tf.cond(
        tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if hparams.pos == "timing":
        targets += timing_signal[:, i:i + 1]
      return targets

    decoder_self_attention_bias = (
      common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
        decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      with tf.variable_scope("body"):
        body_outputs = dp(
          self.dual_decode, targets,
          cache.get("wav_enc_outputs"),
          cache.get("txt_enc_outputs"),
          cache.get("wav_enc_dec_attention_bias"),
          cache.get("txt_enc_dec_attention_bias"),
          bias, hparams, cache,
          nonpadding=features_to_nonpadding(features, "targets"))

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])

      return ret, cache

    ret = fast_decode(
      wav_encoder_output=wav_enc_output,
      txt_encoder_output=txt_enc_output,
      wav_enc_dec_attention_bias=wav_enc_dec_attention_bias,
      txt_enc_dec_attention_bias=txt_enc_dec_attention_bias,
      symbols_to_logits_fn=symbols_to_logits_fn,
      hparams=hparams,
      decode_length=decode_length,
      vocab_size=target_modality.top_dimensionality,
      beam_size=beam_size,
      top_beams=top_beams,
      alpha=alpha,
      batch_size=batch_size,
      force_decode_length=self._decode_hparams.force_decode_length)

    return ret

def transformer_n_encoder(encoder_input,
                          encoder_self_attention_bias,
                          hparams,
                          customize_params,
                          name="encoder",
                          nonpadding=None,
                          save_weights_to=None,
                          make_image_summary=True,
                          losses=None):
  """ transformer with 2 sets of encoders """
  x = encoder_input
  attention_dropout_broadcast_dims = (
    common_layers.comma_separated_string_to_integer_list(
      getattr(hparams, "attention_dropout_broadcast_dims", "")))
  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      padding = common_attention.attention_bias_to_padding(
        encoder_self_attention_bias)
      nonpadding = 1.0 - padding
    pad_remover = None
    if hparams.use_pad_remover and not common_layers.is_on_tpu():
      pad_remover = expert_utils.PadRemover(padding)
    for layer in range(customize_params.num_layers or
                       hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
            common_layers.layer_preprocess(x, hparams),
            None,
            encoder_self_attention_bias,
            hparams.attention_key_channels or hparams.hidden_size,
            hparams.attention_value_channels or hparams.hidden_size,
            hparams.hidden_size,
            customize_params.num_heads or hparams.num_heads,
            hparams.attention_dropout,
            attention_type=hparams.self_attention_type,
            save_weights_to=save_weights_to,
            max_relative_position=hparams.max_relative_position,
            make_image_summary=make_image_summary,
            dropout_broadcast_dims=attention_dropout_broadcast_dims,
            max_length=customize_params.get("max_length"))
          x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
            common_layers.layer_preprocess(x, hparams),
            customized_ffn=customize_params.ffn_layer,
            hparams=hparams,
            pad_remover=pad_remover,
            conv_padding="SAME", nonpadding_mask=nonpadding,
            losses=losses)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def transformer_dual_decoder(decoder_input,
                             wav_encoder_output, txt_encoder_output,
                             decoder_self_attention_bias,
                             wav_enc_dec_attention_bias,
                             txt_enc_dec_attention_bias,
                             hparams,
                             cache=None,
                             name="dual_decoder",
                             nonpadding=None,
                             save_weights_to=None,
                             make_image_summary=True,
                             losses=None):
  """A stack of transformer layers.
  decoder with two attentive interaction with each encoder
  Args:
    decoder_input: a Tensor
    wav_encoder_output: a Tensor
    txt_encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    wav_enc_dec_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    txt_enc_dec_attention_bias: the same as former
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used
      to mask out padding in convolutional layers.  We generally only
      need this mask for "packed" datasets, because for ordinary datasets,
      no padding is ever followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses

  Returns:
    y: a Tensors
  """
  x = decoder_input
  x1 = None
  x2 = None
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  with tf.variable_scope(name):
    for layer in range(hparams.num_decoder_layers or
                       hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          # decoder self-attention
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              save_weights_to=save_weights_to,
              max_relative_position=hparams.max_relative_position,
              cache=layer_cache,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_target_seq_length")) #
          x = common_layers.layer_postprocess(x, y, hparams)
        if wav_encoder_output is not None:
          with tf.variable_scope("wav_encdec_attention"):
            y1 = common_attention.multihead_attention(
                common_layers.layer_preprocess(x, hparams),
                wav_encoder_output,
                wav_enc_dec_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.wav_num_heads or hparams.num_heads,
                hparams.attention_dropout,
                save_weights_to=save_weights_to,
                make_image_summary=make_image_summary,
                dropout_broadcast_dims=attention_dropout_broadcast_dims,
                max_length=hparams.get("max_length") ) #"max_wav_seq_length")*80
            x1 = common_layers.layer_postprocess(x, y1, hparams)
        if txt_encoder_output is not None:
          with tf.variable_scope("txt_encdec_attention"):
            y2 = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              txt_encoder_output,
              txt_enc_dec_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.txt_num_heads or hparams.num_heads,
              hparams.attention_dropout,
              save_weights_to=save_weights_to,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_txt_seq_length") )#max_txt_seq_length
            x2 = common_layers.layer_postprocess(x, y2, hparams)
        with tf.variable_scope("ffn"):
          if wav_encoder_output is not None and txt_encoder_output is not None:
            # with two encoder to attend to
            y = transformer_ffn_layer(
              common_layers.layer_preprocess(tf.concat([x1,x2],axis=-1),
                                             hparams),
              hparams,
              conv_padding="LEFT",
              nonpadding_mask=nonpadding,
              losses=losses,
              cache=layer_cache)
          else:
            y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams),
              hparams,
              conv_padding="LEFT",
              nonpadding_mask=nonpadding,
              losses=losses,
              cache=layer_cache)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)

def fast_decode(wav_encoder_output,
                txt_encoder_output,
                wav_enc_dec_attention_bias,
                txt_enc_dec_attention_bias,
                symbols_to_logits_fn,
                hparams,
                decode_length,
                vocab_size,
                beam_size=1,
                top_beams=1,
                alpha=1.0,
                eos_id=beam_search.EOS_ID,
                batch_size=None,
                force_decode_length=False):
  """ implement greedy and beam search
  Args:
    wav_encoder_output: Output from wav encoder.
    txt_encoder_output: Output from txt encoder.
    wav_enc_dec_attention_bias: a bias tensor for use in enc-dec attention
      over wav inputs
    txt_enc_dec_attention_bias: a bias tensor for use in enc-dec attention
      over txt inputs
    symbols_to_logits_fn: Incremental decoding; function mapping triple
      `(ids, step, cache)` to symbol logits.
    hparams: run hyperparameters
    decode_length: an integer.  How many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    beam_size: number of beams.
    top_beams: an integer. How many of the beams to return.
    alpha: Float that controls the length penalty. larger the alpha, stronger
      the preference for longer translations.
    eos_id: End-of-sequence symbol in beam search.
    batch_size: an integer scalar - must be passed if there is no input
    force_decode_length: bool, whether to force the full decode length, or if
      False, stop when all beams hit eos_id.

  Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if top_beams == 1 or
              [batch_size, top_beams, <= decode_length] otherwise
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If beam size > 1 with partial targets.
  """
  if wav_encoder_output is not None:
    batch_size = common_layers.shape_list(wav_encoder_output)[0]

  key_channels = hparams.attention_key_channels or hparams.hidden_size
  value_channels = hparams.attention_value_channels or hparams.hidden_size
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers

  cache = {
      "layer_%d" % layer: {
          "k": tf.zeros([batch_size, 0, key_channels]),
          "v": tf.zeros([batch_size, 0, value_channels]),
          "f": tf.zeros([batch_size, 0, hparams.hidden_size]),
      } for layer in range(num_layers)
  }

  if txt_encoder_output and wav_encoder_output:
    cache["wav_enc_output"] = wav_encoder_output
    cache["txt_enc_output"] = txt_encoder_output
    cache["wav_enc_dec_attention_bias"] = wav_enc_dec_attention_bias
    cache["txt_enc_dec_attention_bias"] = txt_enc_dec_attention_bias

  if beam_size > 1:  # Beam Search
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)
    decoded_ids, scores = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        eos_id=eos_id,
        stop_early=(top_beams == 1))

    if top_beams == 1:
      decoded_ids = decoded_ids[:, 0, 1:]
      scores = scores[:, 0]
    else:
      decoded_ids = decoded_ids[:, :top_beams, 1:]
      scores = scores[:, :top_beams]
  else: # Greedy search
    # pass
    def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
      """One step of greedy decoding."""
      logits, cache = symbols_to_logits_fn(next_id, i, cache)
      log_probs = common_layers.log_prob_from_logits(logits)
      temperature = (0.0 if hparams.sampling_method == "argmax" else
                     hparams.sampling_temp)
      next_id = common_layers.sample_with_temperature(logits, temperature)
      hit_eos |= tf.equal(next_id, eos_id)

      log_prob_indices = tf.stack(
          [tf.range(tf.to_int64(batch_size)), next_id], axis=1)
      log_prob += tf.gather_nd(log_probs, log_prob_indices)

      next_id = tf.expand_dims(next_id, axis=1)
      decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
      return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

    def is_not_finished(i, hit_eos, *_):
      finished = i >= decode_length
      if not force_decode_length:
        finished |= tf.reduce_all(hit_eos)
      return tf.logical_not(finished)

    decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
    hit_eos = tf.fill([batch_size], False)
    next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
    initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)
    _, _, _, decoded_ids, _, log_prob = tf.while_loop(
        is_not_finished,
        inner_loop, [
            tf.constant(0), hit_eos, next_id, decoded_ids, cache,
            initial_log_prob
        ],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            nest.map_structure(beam_search.get_state_shape_invariants, cache),
            tf.TensorShape([None]),
        ])
    scores = log_prob

  return {"outputs": decoded_ids, "scores": scores}

def transformer_ffn_layer(x,
                          hparams,
                          customized_ffn=None,
                          pad_remover=None,
                          conv_padding="LEFT",
                          nonpadding_mask=None,
                          losses=None,
                          cache=None):
  """Feed-forward layer in the transformer.

  Args:
    x: a Tensor of shape [batch_size, length, hparams.hidden_size]
    hparams: hyperparameters for model
    customized_ffn: customized the ffn_layer string
    pad_remover: an expert_utils.PadRemover object tracking the padding
      positions. If provided, when using convolutional settings, the padding
      is removed before applying the convolution, and restored afterward. This
      can give a significant speedup.
    conv_padding: a string - either "LEFT" or "SAME".
    nonpadding_mask: an optional Tensor with shape [batch_size, length].
      needed for convolutional layers with "SAME" padding.
      Contains 1.0 in positions corresponding to nonpadding.
    losses: optional list onto which to append extra training losses
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.

  Returns:
    a Tensor of shape [batch_size, length, hparams.hidden_size]

  Raises:
    ValueError: If losses arg is None, but layer generates extra losses.
  """
  ffn_layer = customized_ffn or hparams.ffn_layer
  relu_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "relu_dropout_broadcast_dims", "")))
  if ffn_layer == "conv_hidden_relu":
    # Backwards compatibility
    ffn_layer = "dense_relu_dense"
  if ffn_layer == "dense_relu_dense":
    # In simple convolution mode, use `pad_remover` to speed up processing.
    if pad_remover:
      original_shape = common_layers.shape_list(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = common_layers.dense_relu_dense(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        dropout=hparams.relu_dropout,
        dropout_broadcast_dims=relu_dropout_broadcast_dims)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output
  elif ffn_layer == "conv_relu_conv":
    return common_layers.conv_relu_conv(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        first_kernel_size=hparams.conv_first_kernel,
        second_kernel_size=1,
        padding=conv_padding,
        nonpadding_mask=nonpadding_mask,
        dropout=hparams.relu_dropout,
        cache=cache)
  elif ffn_layer == "parameter_attention":
    return common_attention.parameter_attention(
        x, hparams.parameter_attention_key_channels or hparams.hidden_size,
        hparams.parameter_attention_value_channels or hparams.hidden_size,
        hparams.hidden_size, hparams.filter_size, hparams.num_heads,
        hparams.attention_dropout)
  elif ffn_layer == "conv_hidden_relu_with_sepconv":
    return common_layers.conv_hidden_relu(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        kernel_size=(3, 1),
        second_kernel_size=(31, 1),
        padding="LEFT",
        dropout=hparams.relu_dropout)
  elif ffn_layer == "sru":
    return common_layers.sru(x)
  elif ffn_layer == "local_moe_tpu":
    overhead = (hparams.moe_overhead_train
                if hparams.mode == tf.estimator.ModeKeys.TRAIN
                else hparams.moe_overhead_eval)
    ret, loss = expert_utils.local_moe_tpu(
        x, hparams.filter_size // 2,
        hparams.hidden_size,
        hparams.moe_num_experts, overhead=overhead,
        loss_coef=hparams.moe_loss_coef)
    if losses is None:
      raise ValueError(
          "transformer_ffn_layer with type local_moe_tpu must pass in "
          "a losses list")
    losses.append(loss)
    return ret
  else:
    assert ffn_layer == "none"
    return x

@registry.register_hparams
def dual_transformer_nst():
  """ set of hyperparameters. """
  hparams = common_hparams.basic_params1() # training parameters
  hparams.norm_type = "layer"
  hparams.hidden_size = 384
  hparams.batch_size = 8000000

  hparams.max_length = 1500000 # this limits inputs[1] * inputs[2] given to transformer parts
  # max_*_seq_length is used to truncate input labels, this also requires more memory
  hparams.add_hparam("max_wav_seq_length", 0)# falls back to max_length in attention if not set
  hparams.add_hparam("max_txt_seq_length", 2500)# falls back to max_length in attention if not set
  hparams.max_target_seq_length = 0
  hparams.add_hparam("concat_frame", 4)# 0 means no frame-concat

  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.15
  hparams.learning_rate_warmup_steps = 10000
  hparams.initializer_gain = 1.0
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  # hparams.learning_rate_schedule = (
  #     "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size")
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.optimizer_adam_beta1 = 0.9
  # hparams.optimizer_adam_beta2 = 0.997
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.1
  hparams.shared_embedding_and_softmax_weights = True
  hparams.symbol_modality_num_shards = 1
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"

  hparams.daisy_chain_variables = False  # data parallelism

# Add new ones like this.
  hparams.add_hparam("filter_size", 1536)
  # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
  hparams.num_hidden_layers = 4
  hparams.add_hparam("num_wav_enc_layers", 0)
  hparams.add_hparam("num_txt_enc_layers", 6)
  hparams.add_hparam("num_decoder_layers", 0)

  # Attention-related flags.
  hparams.add_hparam("num_heads", 6)
  # multi-head attention setings, falls back to num_heads if 0
  hparams.add_hparam("wav_num_heads", 3)
  hparams.add_hparam("txt_num_heads", 0)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("ffn_layer", "conv_relu_conv")
  hparams.add_hparam("parameter_attention_key_channels", 0)
  hparams.add_hparam("parameter_attention_value_channels", 0)

  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.layer_prepostprocess_dropout_broadcast_dims="1"
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.2
  hparams.add_hparam("attention_dropout_broadcast_dims", "0,1")
  hparams.add_hparam("relu_dropout_broadcast_dims", "1")
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("proximity_bias", False)
  hparams.add_hparam("use_pad_remover", True)
  hparams.add_hparam("self_attention_type", "dot_product")
  hparams.add_hparam("max_relative_position", 0)
  hparams.add_hparam("conv_first_kernel", 9)
  # These parameters are only used when ffn_layer=="local_moe_tpu"
  hparams.add_hparam("moe_overhead_train", 1.0)
  hparams.add_hparam("moe_overhead_eval", 2.0)
  hparams.moe_num_experts = 16
  hparams.moe_loss_coef = 1e-3

  return hparams