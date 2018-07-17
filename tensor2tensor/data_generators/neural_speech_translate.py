# coding=utf-8
import fnmatch
import os
import tarfile

import numpy as np
import tensorflow as tf

from speech_recognition import AudioEncoder
from speech_recognition import add_delta_deltas
from speech_recognition import compute_mel_filterbank_features
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

_NST_ASR_TRAIN_DATASETS = [
    [ # training asr wav files
        "http://www.openslr.org/resources/12/train-clean-100.tar.gz",  # pylint: disable=line-too-long
        "train-clean-100"
    ],
    [
        "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "train-clean-360"
    ],
    [
        "http://www.openslr.org/resources/12/train-other-500.tar.gz",
        "train-other-500"
    ],
]
_NST_ASR_DEV_DATASETS = [
    [ # dev asr wav files and corresponding txt file
        "http://www.openslr.org/resources/12/dev-clean.tar.gz",
        "dev-clean"
    ],
    [
        "http://www.openslr.org/resources/12/dev-other.tar.gz",
        "dev-other"
    ],
]
_NST_ASR_TEST_DATASETS = [
    [
        "http://www.openslr.org/resources/12/test-clean.tar.gz",
        "test-clean"
    ],
    [
        "http://www.openslr.org/resources/12/test-other.tar.gz",
        "test-other"
    ],
]

def _collect_data(directory, in_wav_ext, in_txt_ext, trg_ext):
  """traverse directory to collect wav-txt inputs and target files"""
  # Directory from string to triplets of strings
  # key: the filepath to a datafile including the datafile's basename
  #   if the datafile was "/path/to/datafile.wav" then the key would be
  #   "/path/to/datafile" ( *.wav, *trans.txt, *tmp.txt;)
  # value: a triplet of strings (media_filepath, label)
  data_files = dict()
  for root, _, filenames in os.walk(directory):
    in_txts = [filename for filename in filenames
           if in_txt_ext in filename]
    trg_txts = [filename for filename in filenames
           if trg_ext in filename]

    for in_txt,trg_txt in zip(in_txts, trg_txts):
      in_txt_path = os.path.join(root, in_txt)
      trg_txt_path = os.path.join(root, trg_txt)
      with open(in_txt_path, "r") as in_txt_file, \
        open(trg_txt_path, "r") as trg_txt_file :
        for line in in_txt_file:
          target = trg_txt_file.readline()
          line_content = line.strip().split(" ", 1)
          media_base, txt_input = line_content
          key = os.path.join(root, media_base)
          assert key not in data_files
          media_name = "%s.%s" % (media_base, in_wav_ext)
          media_path = os.path.join(root, media_name)
          data_files[key] = (media_base, media_path, txt_input, target)
  return data_files

class VocabType(object):
  CHARACTER = "character"
  SUBWORD = "subwords"
  TOKEN = "tokens"

@registry.register_problem() #
class NeuralSpeechTranslate(problem.Problem):
  TRAIN_DATASETS = _NST_ASR_TRAIN_DATASETS
  DEV_DATASETS = _NST_ASR_DEV_DATASETS
  TEST_DATASETS = _NST_ASR_TEST_DATASETS

  @property
  def has_inputs(self):
    return True

  @property
  def packed_length(self):
    return None

  @property
  def vocab_type(self):
    return VocabType.SUBWORD

  @property
  def approx_vocab_size(self):
    return 2**15

  @property
  def target_vocab_name(self):
    return "vocab.nst-zh.%d" % self.approx_vocab_size

  @property
  def source_vocab_name(self):
    return "vocab.nst-en.%d" % self.approx_vocab_size

  @property
  def num_shards(self):
    return 100

  @property
  def batch_size_means_tokens(self):
    return True

  @property
  def num_dev_shards(self):
    return 1

  @property
  def num_test_shards(self):
    return 1

  @property
  def use_train_shards_for_dev(self):
    """If true, we only generate training data and hold out shards for dev."""
    return False

  @property
  def output_space_id(self):
    return problem.SpaceID.ZH_TOK

#---------------- for problem t2t-datagen by Vergilus   ------------------------
  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    src_encoder = text_encoder.SubwordTextEncoder(src_vocab_filename)
    trg_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    trg_encoder = text_encoder.SubwordTextEncoder(trg_vocab_filename)
    return {
        "inputs": None,  # Put None to make sure that the logic in
                         # decoding.py doesn't try to convert the floats
                         # into text...
        "txt_inputs": src_encoder,
        "wav_inputs": AudioEncoder(),
        "targets": trg_encoder,
    }

  def get_feature_encoders(self, data_dir=None):
    if self._encoders is None:
      assert data_dir is not None, "dictionary directory must be provided!"
      self._encoders = self.feature_encoders(data_dir)
    return self._encoders

  def compile_data(self, tmp_dir, mode, file_name, file_ext=None):
    """collect data within certain mode
    Args:
      file_name: concate all samples under region and saved to 'path+filename'
      file_ext: file name extension to indicate the src and trg
      mode: string indicating data region:"train" "dev" or "test"
    return: saved filename
    """
    file_name = os.path.join(tmp_dir, file_name)
    if os.path.exists(file_name):
      print "Skipping compile data, found files:%s" % (file_name)
    else:
      # compile data
      with open(file_name, "wb") as collect_samples:
        for root, _, filenames in os.walk(tmp_dir):
          if mode in root:
            for filename in fnmatch.filter(filenames, "*"+file_ext):
              filename = root + '/' + filename
              with open(filename, "r") as input_file:
                for line in input_file:
                  if " "in line.strip():
                    collect_samples.write(line[line.find(" "):].lower().strip() + '\n')
                  else:
                    collect_samples.write(line.strip()+'\n')
      # print file_name+" collected!"
    return file_name

  def get_or_generate_vocab(self, data_dir, tmp_dir=None):

    filename_base = "nst_enzh_%sk_tok_%s" % (self.approx_vocab_size, "train")
    """collect training Files to be passed to generate vocab(src)"""
    src_file_name=self.compile_data(tmp_dir, "train", filename_base+".lang1", file_ext="trans.txt")
    """collect training Files to be passed to generate vocab(trg)"""
    trg_file_name=self.compile_data(tmp_dir, "train", filename_base+".lang2", file_ext="tmp.txt" )

    def traverse_samples(filepath, file_byte_budget):
      with tf.gfile.GFile(filepath, mode="r") as source_file:
        file_byte_budget_ = file_byte_budget
        counter = 0
        countermax = int(source_file.size() / file_byte_budget_ / 2)
        for line in source_file:
          if counter < countermax:
            counter += 1
          else:
            if file_byte_budget_ <= 0:
              break
            line = line.strip()
            file_byte_budget_ -= len(line)
            counter = 0
            yield line
    src_vocab = generator_utils.get_or_generate_vocab_inner(
      data_dir,
      self.source_vocab_name, self.approx_vocab_size,
      traverse_samples(src_file_name, file_byte_budget=1e8),
    )
    trg_vocab = generator_utils.get_or_generate_vocab_inner(
      data_dir,
      self.target_vocab_name, self.approx_vocab_size,
      traverse_samples(trg_file_name, file_byte_budget=1e8),
    )

    return src_vocab, trg_vocab

  def generator(self,data_dir, tmp_dir, datasets,
                eos_list=None, start_from=0, how_many=0):
    del eos_list
    i = 0
    # collect samples and get vocabulary
    src_vocab, trg_vocab=self.get_or_generate_vocab(data_dir, tmp_dir)
    print "src_vocab_size:",src_vocab.vocab_size, "trg_vocab_size:", trg_vocab.vocab_size
    self._encoders = self.feature_encoders(data_dir)

    # unpack data and using encoded samples
    for url, subdir in datasets:
      filename = os.path.basename(url)
      compressed_file = generator_utils.maybe_download(tmp_dir, filename, url)
      read_type = "r:gz" if filename.endswith("tgz") else "r"
      with tarfile.open(compressed_file, read_type) as corpus_tar:
        # Create a subset of files that don't already exist.
        #  tarfile.extractall errors when encountering an existing file
        #  and tarfile.extract is extremely slow
        members = []
        for f in corpus_tar:
          if not os.path.isfile(os.path.join(tmp_dir, f.name)):
            members.append(f)
        corpus_tar.extractall(tmp_dir, members=members)

      tmp_data_dir = os.path.join(tmp_dir, "LibriSpeech", subdir)
      # here we need extra translated results from other src: saved in *.tmp.txt
      data_files = _collect_data(tmp_data_dir, "flac", "trans.txt", "tmp.txt")
      data_triplets = data_files.values()

      audio_encoder = self._encoders["wav_inputs"]
      # turn to external dictionary for txt encoding and generate encoded samples
      for utt_id, wav_input, txt_input, target in sorted(data_triplets)[start_from:]:
        if how_many > 0 and i == how_many:
          return
        i += 1
        wav_data = audio_encoder.encode(wav_input)
        spk_id, unused_book_id, _ = utt_id.split("-")

        yield {
            "wav_inputs": wav_data,
            "wav_inputs_lens": [len(wav_data)],
            "txt_inputs": src_vocab.encode(txt_input)+[text_encoder.EOS_ID],
            "targets": trg_vocab.encode(target)+[text_encoder.EOS_ID],
            "raw_transcript": [txt_input],
            "utt_id": [utt_id],
            "spk_id": [spk_id],
        }

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(
        data_dir, self.num_dev_shards, shuffled=False)
    test_paths = self.test_filepaths(
        data_dir, self.num_test_shards, shuffled=True)

    generator_utils.generate_files(
      self.generator(data_dir, tmp_dir, self.TEST_DATASETS), test_paths
    )
    if self.use_train_shards_for_dev:
      all_paths = train_paths+dev_paths
      generator_utils.generate_files(
        self.generator(data_dir, tmp_dir, self.TRAIN_DATASETS), all_paths
      )
      generator_utils.shuffle_dataset(all_paths)
    else:
      generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, self.TRAIN_DATASETS), train_paths,
        self.generator(data_dir, tmp_dir, self.DEV_DATASETS), dev_paths
      )
#----------------------------------------------------------------------------

  def max_length(self, model_hparams):
    # max length for the input sequences
    return (self.packed_length or
            super(NeuralSpeechTranslate, self).max_length(model_hparams))

  def hparams(self, defaults, model_hparams):
    p = model_hparams
  # -------------------- for audio encoder ---------------------------------
    # Filterbank extraction in bottom instead of preprocess_example is faster.
    p.add_hparam("audio_preproc_in_bottom", True)
    # The trainer seems to reserve memory for all members of the input dict
    p.add_hparam("audio_keep_example_waveforms", False)
    p.add_hparam("audio_sample_rate", 16000)
    p.add_hparam("audio_preemphasis", 0.97)
    p.add_hparam("audio_dither", 1.0 / np.iinfo(np.int16).max)
    p.add_hparam("audio_frame_length", 25.0)
    p.add_hparam("audio_frame_step", 10.0)
    p.add_hparam("audio_lower_edge_hertz", 20.0)
    p.add_hparam("audio_upper_edge_hertz", 8000.0)
    p.add_hparam("audio_num_mel_bins", 80)
    p.add_hparam("audio_add_delta_deltas", True)
    p.add_hparam("num_zeropad_frames", 250)

    p = defaults
    p.stop_at_eos = int(True)
    # the dictionary is initialize in problem.get_hparams when trained
    source_vocab_size = self._encoders["txt_inputs"].vocab_size
    target_vocab_size = self._encoders["targets"].vocab_size
    p.input_modality = {"wav_inputs": ("audio:speech_recognition_modality", None),
                        "txt_inputs": (registry.Modalities.SYMBOL, source_vocab_size)}
    p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
    if self.vocab_type == VocabType.CHARACTER:
      p.loss_multiplier = 2.0


  def example_reading_spec(self):
    data_fields = {
      "wav_inputs":tf.VarLenFeature(tf.float32),
      "txt_inputs":tf.VarLenFeature(tf.int64),
      "targets":tf.VarLenFeature(tf.int64)
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  def preprocess_example(self, example, mode, hparams):
    p = hparams
    # preprocess example['inputs', 'batch_prediction_key', 'targets' ]
    example["waveforms"] = example["wav_inputs"]
    if p.audio_preproc_in_bottom:
      example["wav_inputs"] = tf.expand_dims(
        tf.expand_dims(example["waveforms"], -1), -1)
    else:
      waveforms = tf.expand_dims(example["waveforms"], 0)
      mel_fbanks = compute_mel_filterbank_features(
        waveforms,
        sample_rate=p.audio_sample_rate,
        dither=p.audio_dither,
        preemphasis=p.audio_preemphasis,
        frame_length=p.audio_frame_length,
        frame_step=p.audio_frame_step,
        lower_edge_hertz=p.audio_lower_edge_hertz,
        upper_edge_hertz=p.audio_upper_edge_hertz,
        num_mel_bins=p.audio_num_mel_bins,
        apply_mask=False)
      if p.audio_add_delta_deltas:
        mel_fbanks = add_delta_deltas(mel_fbanks)
      fbank_size = common_layers.shape_list(mel_fbanks)
      assert fbank_size[0] == 1

      # This replaces CMVN estimation on data
      mean = tf.reduce_mean(mel_fbanks, keepdims=True, axis=1)
      variance = tf.reduce_mean((mel_fbanks - mean) ** 2, keepdims=True, axis=1)
      mel_fbanks = (mel_fbanks - mean) / variance

      # Later models like to flatten the two spatial dims. Instead, we add a
      # unit spatial dim and flatten the frequencies and channels.
      example["wav_inputs"] = tf.concat([
        tf.reshape(mel_fbanks, [fbank_size[1], fbank_size[2], fbank_size[3]]),
        tf.zeros((p.num_zeropad_frames, fbank_size[2], fbank_size[3]))], 0)
    if not p.audio_keep_example_waveforms:
      del example["waveforms"]
    # truncate inputs and targets
    if hparams.max_wav_seq_length > 0:
      example["wav_inputs"] = example["wav_inputs"][:hparams.max_wav_seq_length]
    if hparams.max_txt_seq_length > 0:
      example["txt_inputs"] = example["txt_inputs"][:hparams.max_txt_seq_length]
    if hparams.max_target_seq_length > 0:
      example["targets"] = example["targets"][:hparams.max_target_seq_length]

    return example

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
        metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.NEG_LOG_PERPLEXITY,
        metrics.Metrics.APPROX_BLEU, metrics.Metrics.ROUGE_2_F,
        metrics.Metrics.ROUGE_L_F
    ]