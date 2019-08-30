from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import logging
from architectures import biggan_deep
from architectures import arch_ops as ops
from architectures import invertible_network
import tensorflow as tf
import gin


@gin.configurable
class BigGanDeepResNetBlock(biggan_deep.BigGanDeepResNetBlock):
    """ResNet block with bottleneck and identity preserving skip connections."""
    def __init__(self, **kwargs):
        super(BigGanDeepResNetBlock, self).__init__(**kwargs)


@gin.configurable
class Generator(biggan_deep.Generator):
    """ResNet-based generator supporting resolutions 32, 64, 128, 256, 512."""
    def __init__(self, **kwargs):
        """Constructor for BigGAN generator."""
        super(Generator, self).__init__(**kwargs)

    def invert_net(self, z):
        invert_net = invertible_network.Invertible_network(
            in_z_shape=z.shape, in_y_shape=None
        )
        return invert_net(z=z, y=None)

    def apply(self, z, y, is_training):
        """Build the generator network for the given inputs.

        Args:
          z: `Tensor` of shape [batch_size, z_dim] with latent code.
          y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
            labels.
          is_training: boolean, are we in train or eval model.

        Returns:
          A tensor of size [batch_size] + self._image_shape with values in [0, 1].
        """
        shape_or_none = lambda t: None if t is None else t.shape
        logging.info("[Generator] inputs are z=%s, y=%s", z.shape, shape_or_none(y))
        seed_size = 4

        z = self.invert_net(z)

        if self._embed_y:
            y = ops.linear(y, self._embed_y_dim, scope="embed_y", use_sn=False,
                           use_bias=False)
        if y is not None:
            y = tf.concat([z, y], axis=1)
            z = y

        in_channels, out_channels = self._get_in_out_channels()
        num_blocks = len(in_channels)

        # Map noise to the actual seed.
        net = ops.linear(
            z,
            in_channels[0] * seed_size * seed_size,
            scope="fc_noise",
            use_sn=self._spectral_norm)
        # Reshape the seed to be a rank-4 Tensor.
        net = tf.reshape(
            net,
            [-1, seed_size, seed_size, in_channels[0]],
            name="fc_reshaped")

        for block_idx in range(num_blocks):
            scale = "none" if block_idx % 2 == 0 else "up"
            block = self._resnet_block(
                name="B{}".format(block_idx + 1),
                in_channels=in_channels[block_idx],
                out_channels=out_channels[block_idx],
                scale=scale)
            net = block(net, z=z, y=y, is_training=is_training)
            # At resolution 64x64 there is a self-attention block.
            if scale == "up" and net.shape[1].value == 64:
                logging.info("[Generator] Applying non-local block to %s", net.shape)
                net = ops.non_local_block(net, "non_local_block",
                                          use_sn=self._spectral_norm)
        # Final processing of the net.
        # Use unconditional batch norm.
        logging.info("[Generator] before final processing: %s", net.shape)
        net = ops.batch_norm(net, is_training=is_training, name="final_norm")
        net = tf.nn.relu(net)
        colors = self._image_shape[2]
        if self._experimental_fast_conv_to_rgb:

            net = ops.conv2d(net, output_dim=128, k_h=3, k_w=3,
                             d_h=1, d_w=1, name="final_conv",
                             use_sn=self._spectral_norm)
            net = net[:, :, :, :colors]
        else:
            net = ops.conv2d(net, output_dim=colors, k_h=3, k_w=3,
                             d_h=1, d_w=1, name="final_conv",
                             use_sn=self._spectral_norm)
        logging.info("[Generator] after final processing: %s", net.shape)
        net = (tf.nn.tanh(net) + 1.0) / 2.0
        return net


@gin.configurable
class Discriminator(biggan_deep.Discriminator):
    """ResNet-based discriminator supporting resolutions 32, 64, 128, 256, 512."""

    def __init__(self, **kwargs):
        """Constructor for BigGAN discriminator."""
        super(Discriminator, self).__init__(**kwargs)
