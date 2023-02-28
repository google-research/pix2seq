# coding=utf-8
# Copyright 2022 The Pix2Seq Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The diffusion utils."""
import functools
import math
import tensorflow as tf


def sqrt(x):
  return tf.math.sqrt(x)


class Scheduler(object):
  """Time scheduling and add noise to data."""

  def __init__(self, train_schedule):
    self._time_transform = self.get_time_transform(train_schedule)
    self.sample_ddim = functools.partial(
        self.generate, sampler_name='ddim')
    self.sample_ddpm = functools.partial(
        self.generate, sampler_name='ddpm')

  def get_time_transform(self, schedule_name):
    """Returns time transformation function according to schedule name."""
    if schedule_name.startswith('log@'):
      start, end, reverse = schedule_name.split('@')[1].split(',')
      start, end = float(start), float(end)
      reverse = reverse.lower() == 't' or reverse.lower() == 'true'
      time_transform = lambda t: log_schedule(t, start, end, reverse)
    elif schedule_name.startswith('sigmoid@'):
      start, end, tau = schedule_name.split('@')[1].split(',')
      start, end, tau = float(start), float(end), float(tau)
      time_transform = lambda t: sigmoid_schedule(t, start, end, tau)
    elif schedule_name.startswith('cosine'):
      if '@' in schedule_name:
        start, end, tau = schedule_name.split('@')[1].split(',')
        start, end, tau = float(start), float(end), float(tau)
        time_transform = lambda t: cosine_schedule(t, start, end, tau)
      else:
        time_transform = cosine_schedule_simple
    elif schedule_name.startswith('simple_linear'):
      time_transform = simple_linear_schedule
    else:
      raise ValueError(f'Unknown train schedule {schedule_name}')
    return time_transform

  def time_transform(self, time_step):
    return self._time_transform(time_step)

  def sample_noise(self, shape):
    """Sample noises."""
    return tf.random.normal(shape)

  def add_noise(self,
                inputs,
                time_step=None):
    """Forword process."""
    bsz = tf.shape(inputs)[0]
    ndim = inputs.shape.rank
    time_step_shape = [bsz] + [1] * (ndim - 1)
    if time_step is None:
      time_step = tf.random.uniform(time_step_shape, 0, 1, dtype=tf.float32)
    elif isinstance(time_step, float):
      time_step = tf.ones(time_step_shape, dtype=tf.float32) * time_step
    else:
      time_step = tf.reshape(time_step, time_step_shape)
    gamma = self.time_transform(time_step)
    noise = self.sample_noise(tf.shape(inputs))
    inputs_noised = inputs * sqrt(gamma) + noise * sqrt(1 - gamma)
    return inputs_noised, noise, time_step, gamma

  def transition_step(self,
                      samples,
                      data_pred,
                      noise_pred,
                      gamma_now,
                      gamma_prev,
                      sampler_name):
    """Transition to states with a smaller time step."""
    ddpm_var_type = 'large'
    if sampler_name.startswith('ddpm') and '@' in sampler_name:
      ddpm_var_type = sampler_name.split('@')[1]

    if sampler_name == 'ddim':
      samples = data_pred * sqrt(gamma_prev) + noise_pred * sqrt(1 -
                                                                 gamma_prev)
    elif sampler_name.startswith('ddpm'):
      log_alpha_t = tf.math.log(gamma_now) - tf.math.log(gamma_prev)
      alpha_t = tf.clip_by_value(tf.math.exp(log_alpha_t), 0.0, 1.0)
      x_mean = tf.math.rsqrt(alpha_t) * (
          samples - tf.math.rsqrt(1 - gamma_now) * (1 - alpha_t) * noise_pred)
      if ddpm_var_type == 'large':
        var_t = 1.0 - alpha_t  # var = beta_t
      elif ddpm_var_type == 'small':
        var_t = tf.math.exp(
            tf.math.log1p(-gamma_prev) - tf.math.log1p(-gamma_now)) * (1.0 -
                                                                       alpha_t)
      else:
        raise ValueError(f'Unknown ddpm_var_type {ddpm_var_type}')
      eps = self.sample_noise(tf.shape(data_pred))
      samples = x_mean + tf.math.sqrt(var_t) * eps
    return samples

  def generate(self,
               transition_f,
               iterations,
               samples_shape,
               hidden_shapes=None,
               pred_type='eps',
               schedule=None,
               td=0.,
               x0_clip='',
               self_cond='none',
               self_cond_decay=0.,
               guidance=0.,
               sampler_name='ddim'):
    """A sampling function.

    Args:
      transition_f: `callable` function for producing transition variables.
      iterations: `int` number of iterations for generation.
      samples_shape: `tuple` or `list` shape of samples, e.g., (bsz, h, w, 3).
      hidden_shapes: a `list` of shapes of hiddens from denoising network,
        excluding bsz dim. Set to None if only single tensor output.
      pred_type: `str`, the output type of `transition_f`.
      schedule: `str`, the sampling schedule. If None, use the one during train.
      td: `float` specifying an adjustment for next time step.
      x0_clip: `str` specifying the range of x0 clipping, e.g. '0,1'. Set to
        None or empty str for no clipping.
      self_cond: `str`.
      self_cond_decay: `float` decaying factor between 0 and 1.
      guidance: `float` for cf guidance strength.
      sampler_name: `str`.

    Returns:
      Generated samples in samples_shape.
    """
    num_samples = samples_shape[0]
    ts = tf.ones([num_samples] + [1] * (len(samples_shape) - 1))
    get_step = lambda t: ts * (  # pylint: disable=g-long-lambda
        1.0 - tf.cast(t, tf.float32) / iterations)
    time_transform = self.time_transform if schedule is None else (
        self.get_time_transform(schedule))

    samples = self.sample_noise(samples_shape)
    noise_pred, data_pred = tf.zeros_like(samples), tf.zeros_like(samples)
    x0_clip_fn = get_x0_clipping_function(x0_clip)
    if hidden_shapes is None:
      ctx = SelfCondEstimateContext(
          self_cond, samples_shape, self_cond_decay)
    else:
      ctx = SelfCondHiddenContext(
          self_cond, samples_shape, hidden_shapes, self_cond_decay)
    pred_out = ctx.init_denoise_out(samples_shape)
    for t in tf.range(iterations):
      t = tf.cast(t, tf.float32)
      time_step = get_step(t)
      time_step_p = tf.maximum(get_step(t + 1 + td), 0)
      gamma, gamma_prev = time_transform(time_step), time_transform(time_step_p)
      ctx.update_context({'denoise_out': pred_out,
                          'data_pred': data_pred,
                          'noise_pred': noise_pred,
                          'pred_type': pred_type})
      if guidance == 0:
        pred_out_nl = 0.
      else:
        pred_out_nl = transition_f(
            ctx.contextualized_inputs(samples), gamma, training=False,
            drop_label=True)
        pred_out_nl = pred_out_nl[0] if isinstance(pred_out_nl, tuple) else (
            pred_out_nl)
      pred_out_l = transition_f(
          ctx.contextualized_inputs(samples), gamma, training=False)
      pred_out = pred_out_l
      pred_out_l = pred_out_l[0] if isinstance(pred_out_l, tuple) else (
          pred_out_l)
      pred_out_ = pred_out_l * (1+guidance) - pred_out_nl * guidance
      x0_eps = get_x0_eps(
          samples, gamma, pred_out_, pred_type, x0_clip_fn, truncate_noise=True)
      noise_pred, data_pred = x0_eps['noise_pred'], x0_eps['data_pred']
      samples = self.transition_step(
          samples=samples,
          data_pred=data_pred,
          noise_pred=noise_pred,
          gamma_now=gamma,
          gamma_prev=gamma_prev,
          sampler_name=sampler_name)
    return data_pred


class SelfCondEstimateContext(object):
  """Context manager for self-conditioning estimate during the inference."""

  def __init__(self, mode, samples_shape, momentum=0):
    """Init function.

    Args:
      mode: self-conditioning option.
      samples_shape: `tuple` or `list` shape of samples, e.g., (bsz, h, w, 3).
      momentum: `int` indiciating the momentum averaging for vars.
    """
    self._mode = mode
    self._estimate = tf.zeros(samples_shape)
    self._momentum = momentum

  def init_denoise_out(self, samples_shape):
    """Initial denoising network output for update_context."""
    return tf.zeros(samples_shape)

  def update_context(self, context):
    """Update context / self-conditioning estimate."""
    if self._mode == 'none':
      return

    def ema(v_old, v_new, momentum):
      return momentum * v_old + (1 - momentum) * v_new

    estimate = get_self_cond_estimate(context['data_pred'],
                                      context['noise_pred'],
                                      self._mode,
                                      context['pred_type'])
    self._estimate = ema(self._estimate, estimate, self._momentum)

  def contextualized_inputs(self, samples):
    """Instead of using samples as inputs, obtain inputs with self-cond vars."""
    if self._mode == 'none':
      return samples
    else:
      return tf.concat([samples, self._estimate], axis=-1)


class SelfCondHiddenContext(object):
  """Context manager for self-conditioning variables during the inference."""

  def __init__(self, mode, samples_shape, hidden_shapes, momentum=0):
    """Init function.

    Args:
      mode: self-conditioning option.
      samples_shape: `tuple` or `list` shape of samples, e.g., (bsz, h, w, 3).
      hidden_shapes: a `list` of shapes of extra input/output elements to the
        denoising network, excluding bsz dim. Set to None if not tracked.
      momentum: `int` indiciating the momentum averaging for vars.
    """
    self._mode = mode
    var_shapes = [[samples_shape[0]] + list(s) for s in hidden_shapes]
    self._hiddens = [tf.zeros(shape) for shape in var_shapes]
    self._momentum = momentum

  def init_denoise_out(self, samples_shape):
    """Initial denoising network output for update_context."""
    return tuple([tf.zeros(samples_shape)] + self._hiddens)

  def update_context(self, context):
    """Update context / self-conditioning variables."""
    def ema(v_old, v_new, momentum):
      return momentum * v_old + (1 - momentum) * v_new

    vars_update = []
    hiddens = context['denoise_out'][1:]  # excluding 0-th which is estimate.
    for var_old, var_new in zip(self._hiddens, hiddens):
      vars_update.append(ema(var_old, var_new, self._momentum))
    self._hiddens = vars_update

  def contextualized_inputs(self, samples):
    """Instead of using samples as inputs, obtain inputs with self-cond vars."""
    return tuple([samples] + self._hiddens)


def cosine_schedule_simple(t, ns=0.0002, ds=0.00025):
  """Cosine schedule.

  Args:
    t: `float` between 0 and 1.
    ns: `float` numerator constant shift.
    ds: `float` denominator constant shift.

  Returns:
    `float` of transformed time between 0 and 1.
  """
  return tf.math.cos(((t + ns) / (1 + ds)) * math.pi / 2)**2


def cosine_schedule(t, start=0., end=0.5, tau=1.0, clip_min=1e-9):
  """Cosine schedule.

  Args:
    t: `float` between 0 and 1.
    start: `float` starting point in x-axis of cosine function.
    end: `float` ending point in x-axis of cosine function.
    tau: `float` temperature.
    clip_min: `float` lower bound for output.

  Returns:
    `float` of transformed time between 0 and 1.
  """
  y_start = tf.math.cos(start * math.pi/2)**(2*tau)
  y_end = tf.math.cos(end * math.pi/2)**(2*tau)
  output = (tf.math.cos((t*(end-start)+start)*math.pi/2)**(2*tau)-y_end) / (
      y_start-y_end)
  return tf.clip_by_value(output, clip_min, 1.)


def sigmoid_schedule(t, start=-3., end=3., tau=1.0, clip_min=1e-9):
  """Sigmoid schedule.

  Args:
    t: `float` between 0 and 1.
    start: `float` starting point in x-axis of sigmoid function.
    end: `float` ending point in x-axis of sigmoid function.
    tau: `float` scaling temperature for sigmoid function.
    clip_min: `float` lower bound for output.

  Returns:
    `float` of transformed time between 0 and 1.
  """
  v_start = tf.sigmoid(start / tau)
  v_end = tf.sigmoid(end / tau)
  output = (-tf.sigmoid((t * (end - start) + start) / tau) + v_end) / (
      v_end - v_start)
  return tf.clip_by_value(output, clip_min, 1.)


def log_schedule(t, start=1., end=100., reverse=False):
  """Log schedule.

  Args:
    t: `float` between 0 and 1.
    start: `float` starting point in x-axis of log function.
    end: `float` ending point in x-axis of log function.
    reverse: `boolean` whether to reverse the curving direction.

  Returns:
    `float` of transformed time between 0 and 1.
  """
  if reverse:
    start, end = end, start
  v_start = tf.math.log(start)
  v_end = tf.math.log(end)
  output = (-tf.math.log(t * (end - start) + start) + v_end) / (v_end - v_start)
  return tf.clip_by_value(output, 0., 1.)


def simple_linear_schedule(t, clip_min=1e-9):
  """Simple linear schedule.

  Args:
    t: `float` between 0 and 1.
    clip_min: `float` lower bound for output.

  Returns:
    `float` of transformed time between 0 and 1.
  """
  return tf.clip_by_value(1. - t, clip_min, 1.)


def sample_studentt(shape, df, loc, scale, dtype=tf.float32):
  """Samples from a Student T distribution."""
  normal_sample = tf.random.normal(shape, dtype=dtype)
  gamma_sample = tf.random.gamma(shape, alpha=0.5 * df, beta=0.5)
  samples = normal_sample * tf.math.rsqrt(gamma_sample / df)
  return samples * scale + loc


def get_x0_clipping_function(x0_clip):
  """Str for clip range, eg '0,1'. Set to None or empty str for no clipping."""
  if x0_clip:
    min_v, max_v = x0_clip.split(',')
    x0_clip = lambda x: tf.clip_by_value(x, float(min_v), float(max_v))
  else:
    x0_clip = lambda x: x
  return x0_clip


def get_x0_from_eps(xt, gamma, noise_pred):
  data_pred = 1. / sqrt(gamma) * (xt - sqrt(1. - gamma) * noise_pred)
  return data_pred


def get_eps_from_x0(xt, gamma, data_pred):
  noise_pred = 1. / sqrt(1 - gamma) * (xt - sqrt(gamma) * data_pred)
  return noise_pred


def get_x0_from_v(xt, gamma, v_pred):
  return sqrt(gamma) * xt - sqrt(1 - gamma) * v_pred


def get_eps_from_v(xt, gamma, v_pred):
  return sqrt(1 - gamma) * xt + sqrt(gamma) * v_pred


def get_x0_eps(xt,
               gamma,
               denoise_out,
               pred_type,
               x0_clip_fn,
               truncate_noise=False):
  """Get x0 and eps from denoising output."""
  if pred_type == 'eps':
    noise_pred = denoise_out
    data_pred = get_x0_from_eps(xt, gamma, noise_pred)
    data_pred = x0_clip_fn(data_pred)
    if truncate_noise:
      noise_pred = get_eps_from_x0(xt, gamma, data_pred)
  elif pred_type.startswith('x'):
    data_pred = denoise_out
    data_pred = x0_clip_fn(data_pred)
    noise_pred = get_eps_from_x0(xt, gamma, data_pred)
  elif pred_type.startswith('v'):
    v_pred = denoise_out
    data_pred = get_x0_from_v(xt, gamma, v_pred)
    data_pred = x0_clip_fn(data_pred)
    if truncate_noise:
      noise_pred = get_eps_from_x0(xt, gamma, data_pred)
    else:
      noise_pred = get_eps_from_v(xt, gamma, v_pred)
  else:
    raise ValueError(f'Unknown pred_type {pred_type}')
  return {'noise_pred': noise_pred, 'data_pred': data_pred}


def get_self_cond_estimate(data_pred, noise_pred, self_cond, pred_type):
  """Returns self cond estimate given predicted data or noise."""
  assert self_cond in ['x', 'eps', 'auto']
  if self_cond == 'x':
    estimate = data_pred
  elif self_cond == 'eps':
    estimate = noise_pred
  else:
    estimate = noise_pred if pred_type == 'eps' else data_pred
  return estimate


def add_self_cond_estimate(x_noised,
                           gamma,
                           denoise_f,
                           pred_type,
                           self_cond,
                           x0_clip,
                           num_sc_examples,
                           drop_rate=0.,
                           training=True):
  """Returns x_noised with self cond estimate added for the first 1/2 batch."""
  assert self_cond in ['x', 'eps', 'auto']
  if drop_rate > 0:
    raise NotImplementedError('Self-Cond by masking is not implemented yet!')
  x_noised_p = x_noised[:num_sc_examples]
  gamma_p = gamma[:num_sc_examples]
  placeholder = tf.zeros_like(x_noised_p)
  pred_out = denoise_f(
      tf.concat([x_noised_p, placeholder], -1), gamma_p, training)
  x0_clip_fn = get_x0_clipping_function(x0_clip)
  x0_eps = get_x0_eps(
      x_noised_p, gamma_p, pred_out, pred_type, x0_clip_fn, truncate_noise=True)
  estimate = get_self_cond_estimate(x0_eps['data_pred'], x0_eps['noise_pred'],
                                    self_cond, pred_type)
  estimate = tf.concat([estimate, tf.zeros_like(x_noised[num_sc_examples:])], 0)
  estimate = tf.stop_gradient(estimate)
  return tf.concat([x_noised, estimate], -1)


def add_self_cond_hidden(x_noised,
                         gamma,
                         denoise_f,
                         num_sc_examples,
                         hidden_shapes,
                         drop_rate=0.,
                         training=True):
  """Returns inputs (with self-cond hiddens) to denoising networks."""
  bsz = tf.shape(x_noised)[0]  # assuming bsz > 1
  x_noised_p = x_noised[:num_sc_examples]
  gamma_p = gamma[:num_sc_examples]
  placeholders1 = [tf.zeros([num_sc_examples]+s) for s in hidden_shapes]
  placeholders2 = [tf.zeros([bsz-num_sc_examples]+s) for s in hidden_shapes]
  pred_out = denoise_f(tuple([x_noised_p] + placeholders1), gamma_p, training)
  hiddens = [tf.concat([u, v], 0) for u, v in zip(pred_out[1:], placeholders2)]
  if drop_rate > 0:  # The rate of masking out self-cond hiddens.
    masks = tf.cast(tf.random.uniform([bsz]) > drop_rate, tf.float32)
    expand_dims = lambda x, h: tf.reshape(x, [bsz] + [1] * (h.shape.ndims - 1))
    hiddens = [h * expand_dims(masks, h) for h in hiddens]
  return [x_noised] + [tf.stop_gradient(h) for h in hiddens]
