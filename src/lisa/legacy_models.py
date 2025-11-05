"""Legacy model and preprocessing helpers retained for compatibility."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from scipy.signal import savgol_coeffs


ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,  # Also known as Swish
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'identity': nn.Identity
}

class SavGolFilterGPU(nn.Module):
    """
    Applies Savitzky-Golay filtering along the channel (spectral) dimension
    using GPU-accelerated 1D convolution.

    Assumes input tensor shape like (B, C, H, W) or (C, H, W) or (C,)
    and filters along the C dimension.
    """

    def __init__(self, window_length: int, polyorder: int = 2, deriv: int = 0, delta: float = 1.0, device=None):
        super().__init__()

        if window_length % 2 == 0:
            raise ValueError("window_length must be odd.")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length.")

        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        # self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device if device else torch.device("cpu")

        # Get SavGol coefficients using scipy
        coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)

        # Scipy returns filter centered, Conv1d applies left-aligned. Reverse coeffs.
        coeffs = coeffs[::-1].copy()
        coeffs_tensor = torch.tensor(coeffs, dtype=torch.float32)  # .to(self.device) # Move later

        # Reshape coeffs for nn.Conv1d: (out_channels=1, in_channels/groups=1, kernel_size)
        kernel = coeffs_tensor.view(1, 1, window_length)

        # Create the non-trainable Conv1d layer
        # Padding='same' ensures output length matches input length along the filtered dim
        # Padding mode 'replicate' is often reasonable for spectral boundaries
        self.filter = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=window_length,
            padding='same',  # Output size = Input size (along C dim)
            bias=False,
            padding_mode='replicate'  # Or 'reflect', 'circular'
        )

        # Set the fixed weights
        self.filter.weight = nn.Parameter(kernel, requires_grad=False)

        # Move the filter layer to the target device
        self.filter.to(self.device)
        print(f"SavGol GPU filter initialized on device: {self.device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the filter. Handles different input dimensions.
        Input x shape: (B, C, H, W) or (C, H, W) or (C,)
        Output shape: Matches input shape.
        """
        original_shape = x.shape
        original_device = x.device
        original_dtype = x.dtype
        ndim = x.ndim

        # Ensure input is on the same device as the filter
        x = x.to(self.device, dtype=torch.float32)  # Filter uses float32

        if ndim == 1:  # Input shape (C,)
            c = original_shape[0]
            # Reshape to (N=1, Channels=1, SeqLen=C) for Conv1d
            x_reshaped = x.view(1, 1, c)
        elif ndim == 3:  # Input shape (C, H, W)
            c, h, w = original_shape
            # Reshape to (N=H*W, Channels=1, SeqLen=C)
            x_reshaped = x.permute(1, 2, 0).reshape(-1, 1, c)  # -> (H, W, C) -> (H*W, 1, C)
        elif ndim == 4:  # Input shape (B, C, H, W)
            b, c, h, w = original_shape
            # Reshape to (N=B*H*W, Channels=1, SeqLen=C)
            x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, 1, c)  # -> (B, H, W, C) -> (B*H*W, 1, C)
        else:
            raise ValueError(f"Input tensor ndim must be 1, 3, or 4. Got {ndim}")

        # Apply convolution
        filtered_x_reshaped = self.filter(x_reshaped)  # Output shape (N, 1, C)

        # Reshape back to original structure
        if ndim == 1:
            filtered_x = filtered_x_reshaped.view(c)
        elif ndim == 3:
            filtered_x = filtered_x_reshaped.view(h, w, c).permute(2, 0, 1)  # -> (H, W, C) -> (C, H, W)
        elif ndim == 4:
            filtered_x = filtered_x_reshaped.view(b, h, w, c).permute(0, 3, 1, 2)  # -> (B, H, W, C) -> (B, C, H, W)

        # Return tensor on original device and dtype
        return filtered_x.to(device=original_device, dtype=original_dtype)

class SpectralPredictorWithTransformer(nn.Module):
    def __init__(self, model_config):
        """
        Initializes the Spectral Predictor model, now including Transformer options.

        Args:
            model_config (dict): A dictionary containing model hyperparameters.
                Includes previous keys plus new ones for transformer:
                - model_type (str): 'mlp', 'cnn1d', 'cnn2d', 'transformer', 'cnn_transformer'.
                - num_bands (int): Number of spectral bands.
                - patch_height (int): Height of the input patch.
                - patch_width (int): Width of the input patch.
                # --- CNN specific (used for cnn1d, cnn2d, cnn_transformer) ---
                - shared_cnn_channels (list): Output channels for CNN layers.
                - shared_cnn_kernels (list): Kernel sizes for CNN layers.
                - shared_cnn_pools (list): Pooling sizes for CNN layers.
                - use_batchnorm (bool): Use BatchNorm in CNNs.
                # --- MLP specific (used for mlp) ---
                - shared_mlp_sizes (list): Hidden sizes for shared MLP layers.
                # --- Transformer specific (used for transformer, cnn_transformer) ---
                - transformer_patch_size (int): Size of smaller patches for tokenization (e.g., 4 means 4x4).
                                                Input H/W (or CNN output H/W) must be divisible by this.
                - transformer_embed_dim (int): Embedding dimension for transformer tokens.
                - transformer_depth (int): Number of transformer encoder layers.
                - transformer_heads (int): Number of attention heads.
                - transformer_mlp_dim (int): Dimension of the MLP head within transformer layers.
                - transformer_dropout (float): Dropout rate within transformer layers.
                - use_cls_token (bool): Whether to use a [CLS] token for aggregation. If False, uses mean pooling.
                # --- Common ---
                - regression_hidden_sizes (list): Hidden sizes for the final regression head.
                - dropout_shared (float): Dropout for shared MLP/CNN layers (not Transformer).
                - dropout_regression (float): Dropout for the final regression head.
        """
        super().__init__()
        self.model_config = model_config
        self.model_type = model_config['model_type']
        self.num_bands = model_config['num_bands']
        self.patch_height = model_config['patch_height']
        self.patch_width = model_config['patch_width']
        self.use_pos_embed = model_config.get('use_pos_embed', True)  # Default to True
        self.use_cls_token = model_config.get('use_cls_token', True)  # Default to True
        self.use_multi_head = model_config.get('use_multi_head', False)  # Default to False
        self.instanceNorm = model_config.get('instanceNorm', False)
        self.AE = model_config.get('AE', False)  # Default to False
        self.conv_dim = model_config.get('conv_dim', 2)
        self.discriminator = model_config.get('discriminator', 0)
        encoder_output_shape = model_config.get('encoder_output_shape', [])
        if encoder_output_shape:
            self.encoder_C_deep = encoder_output_shape[0 if self.conv_dim == 2 else 1]  # Default to 0
            self.encoder_H = encoder_output_shape[1 if self.conv_dim == 2 else 2]  # Default to 0
            self.encoder_W = encoder_output_shape[2 if self.conv_dim == 2 else 3]  # Default to 0

        self.cls = model_config.get('cls', False)  # Default to False
        # --- Build Model Components ---
        self.t_embed_dim = None
        self.t_nheads = None
        self.cnn_backbone = None
        self.patch_embed = None
        self.transformer_encoder = None
        self.pos_embed = None
        self.cls_token = None
        self.shared_layers = None  # For non-transformer models
        self.regression_head = None
        self.t_patch_size = None
        current_h, current_w, current_c = self.patch_height, self.patch_width, self.num_bands
        in_features_regressor = 0
        self.activation_type = model_config.get('activation_type', 'relu').lower()  # Default to relu
        if self.activation_type not in ACTIVATION_MAP:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")
        self.activation_layer = ACTIVATION_MAP[self.activation_type]

        # 1. --- Optional CNN Backbone (for cnn_transformer) ---
        if self.model_type == 'cnn_transformer':
            if not all(k in model_config for k in ['shared_cnn_channels', 'shared_cnn_kernels', 'shared_cnn_pools']):
                raise ValueError("CNN parameters required for cnn_transformer model type.")
            self.cnn_backbone = self._build_cnn_backbone('cnn2d')  # Use 2D CNN for backbone
            # Calculate the output dimensions of the CNN backbone
            current_c, current_h, current_w = self._calculate_cnn_output_dim(self.cnn_backbone)
            print(f"CNN Backbone Output Dim: C={current_c}, H={current_h}, W={current_w}")
 
        # 2. --- Patch Embedding (for transformer types) ---
        if self.model_type in ['transformer', 'cnn_transformer']:
            self.t_patch_size = model_config['transformer_patch_size']
            if current_h % self.t_patch_size != 0 or current_w % self.t_patch_size != 0:
                for new_patch_size in range(self.t_patch_size, 0, -1):
                    if current_h % new_patch_size == 0 and current_w % new_patch_size == 0:
                        self.t_patch_size = new_patch_size
                        print(
                            f"Adjusted patch size to {self.t_patch_size} for input dimensions ({current_h}, {current_w})")
                        break

            # self.num_patches = (current_h // self.t_patch_size) * (current_w // self.t_patch_size)
            num_patches_h = math.floor((current_h - self.t_patch_size) / model_config['transformer_stride']) + 1
            num_patches_w = math.floor((current_w - self.t_patch_size) / model_config['transformer_stride']) + 1
            self.num_patches = num_patches_h * num_patches_w
            patch_dim = current_c * self.t_patch_size * self.t_patch_size  # Flattened size of one sub-patch

            self.t_embed_dim = model_config['transformer_embed_dim']
            self.t_nheads = model_config['transformer_heads']
            if self.t_embed_dim % self.t_nheads != 0:
                print(f"Embedding dimension {self.t_embed_dim} must be divisible by number of heads {self.t_nheads}.")
                self.t_nheads = self.t_embed_dim // 2  # Adjust to be divisible by 2
            # Efficient patching and projection using a Conv2d
            self.patch_embed = nn.Conv2d(current_c, self.t_embed_dim,
                                         kernel_size=self.t_patch_size, stride=model_config['transformer_stride'])

            # --- Positional Embedding & CLS Token ---
            if self.use_cls_token and self.use_pos_embed:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.t_embed_dim))
                # Add 1 to sequence length for CLS token
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.t_embed_dim))
                nn.init.trunc_normal_(self.pos_embed, std=.02)  # Initialize positional embedding
                nn.init.trunc_normal_(self.cls_token, std=.02)
            elif self.use_pos_embed:
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.t_embed_dim))
                nn.init.trunc_normal_(self.pos_embed, std=.02)

            # --- Transformer Encoder ---
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.t_embed_dim,
                nhead=self.t_nheads,
                dim_feedforward=model_config['transformer_mlp_dim'],
                dropout=model_config['transformer_dropout'],
                activation=self.activation_type,  # or 'gelu'
                batch_first=True  # Important: Input shape (B, Seq, Emb)
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=model_config['transformer_depth']
            )
            in_features_regressor = self.t_embed_dim  # Input to regressor is the transformer output dim

        # 3. --- Standard MLP/CNN Shared Layers (for non-transformer types) ---
        elif self.model_type == 'mlp':
            self.input_flattened_size = self.num_bands * self.patch_height * self.patch_width
            in_features = self.input_flattened_size
            shared_layers_list = []
            activation_layer_mlp = self.activation_layer()

            for hidden_size in model_config.get('shared_mlp_sizes', []):
                shared_layers_list.append(nn.Linear(in_features, hidden_size))
                shared_layers_list.append(activation_layer_mlp)
                shared_layers_list.append(nn.Dropout(p=model_config['dropout_shared']))
                in_features = hidden_size
            self.shared_layers = nn.Sequential(*shared_layers_list)
            in_features_regressor = in_features  # Input to regressor is output of last shared layer

        elif self.model_type in ['cnn1d', 'cnn2d']:
            self.shared_layers = self._build_cnn_backbone(self.model_type)
            # Calculate flatten size after CNN layers
            final_channels, final_h, final_w = self._calculate_cnn_output_dim(self.shared_layers)
            if self.model_type == 'cnn1d':
                in_features_regressor = final_channels * final_w  # For 1D CNN output (C, SeqLen)
            else:  # cnn2d
                in_features_regressor = final_channels * final_h * final_w

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # 4. --- Regression Head 1 ---
        reg_layers1 = []
        in_features1 = in_features_regressor
        activation_layer_reg1 = self.activation_layer()

        for hidden_size in model_config['regression_hidden_sizes']:
            reg_layers1.append(nn.Dropout(p=model_config['dropout_regression']))
            reg_layers1.append(nn.Linear(in_features1, hidden_size))
            reg_layers1.append(activation_layer_reg1)
            in_features1 = hidden_size
        reg_layers1.append(nn.Dropout(p=model_config['dropout_regression']))
        reg_layers1.append(nn.Linear(in_features1, 1 if not self.discriminator else 2))
        self.regression_head1 = nn.Sequential(*reg_layers1)

        # 5. --- Regression Head 2 ---
        reg_layers2 = []
        in_features2 = in_features_regressor
        activation_layer_reg2 = self.activation_layer()

        for hidden_size in model_config['regression_hidden_sizes']:
            reg_layers2.append(nn.Dropout(p=model_config['dropout_regression']))
            reg_layers2.append(nn.Linear(in_features2, hidden_size))
            reg_layers2.append(activation_layer_reg2)
            in_features2 = hidden_size
        reg_layers2.append(nn.Dropout(p=model_config['dropout_regression']))
        reg_layers2.append(nn.Linear(in_features2, 1))
        self.regression_head2 = nn.Sequential(*reg_layers2)
        if self.cls:
            # 6. --- Classification Head 1 ---
            cls_layers = []
            in_features1 = in_features_regressor
            activation_layer_cls = self.activation_layer()

            for hidden_size in model_config['cls_hidden_sizes']:
                cls_layers.append(nn.Dropout(p=model_config['dropout_regression']))
                cls_layers.append(nn.Linear(in_features1, hidden_size))
                cls_layers.append(activation_layer_cls)
                in_features1 = hidden_size
            cls_layers.append(nn.Dropout(p=model_config['dropout_regression']))
            cls_layers.append(nn.Linear(in_features1, 2))
            self.cls_head = nn.Sequential(*cls_layers)

    def _build_cnn_backbone(self, cnn_type):
        """Helper to build CNN layers (1D or 2D)."""
        layers = []
        activation_layer = self.activation_layer()  # Instantiate the layer
        if cnn_type == 'cnn1d':
            in_channels = self.num_bands
            current_seq_len = self.patch_height * self.patch_width
            cnn_channels = self.model_config['shared_cnn_channels']
            cnn_kernels = self.model_config['shared_cnn_kernels']
            cnn_pools = self.model_config['shared_cnn_pools']

            for out_channels, kernel_size, pool_size in zip(cnn_channels, cnn_kernels, cnn_pools):
                padding = kernel_size // 2
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
                if self.model_config.get('use_batchnorm', True):
                    batchNorm = nn.BatchNorm1d(out_channels) if not self.instanceNorm else nn.InstanceNorm1d(out_channels)
                    layers.append(batchNorm)
                layers.append(activation_layer)
                if self.model_config['dropout_shared'] > 0:
                    layers.append(nn.Dropout(p=self.model_config['dropout_shared']))
                if pool_size > 1:
                    layers.append(nn.MaxPool1d(kernel_size=pool_size))
                in_channels = out_channels
        elif cnn_type == 'cnn2d':
            in_channels = self.num_bands
            cnn_channels = self.model_config['shared_cnn_channels']
            cnn_kernels = self.model_config['shared_cnn_kernels']  # Expect tuples (kh, kw) or int
            cnn_pools = self.model_config['shared_cnn_pools']  # Expect tuples (ph, pw) or int

            for out_channels, kernel_size, pool_size in zip(cnn_channels, cnn_kernels, cnn_pools):
                if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
                if isinstance(pool_size, int): pool_size = (pool_size, pool_size)
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)

                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
                if self.model_config.get('use_batchnorm', True):
                    batchNorm = nn.BatchNorm2d(out_channels) if not self.instanceNorm else nn.InstanceNorm2d(out_channels)
                    layers.append(batchNorm)
                layers.append(activation_layer)
                if self.model_config['dropout_shared'] > 0:
                    layers.append(nn.Dropout(p=self.model_config['dropout_shared']))  # or Dropout2d
                if pool_size[0] > 1 or pool_size[1] > 1:
                    layers.append(nn.MaxPool2d(kernel_size=pool_size))
                in_channels = out_channels
        else:
            raise ValueError("cnn_type must be 'cnn1d' or 'cnn2d'")

        return nn.Sequential(*layers)

    def _calculate_cnn_output_dim(self, cnn_module):
        """Calculates CNN output C, H, W using a dummy forward pass."""
        cnn_module.eval()  # Important for deterministic output size
        with torch.no_grad():
            # Create dummy input (B=1, C, H, W) or (B=1, C, L)
            if isinstance(cnn_module[0], nn.Conv2d):  # First layer tells us type
                dummy_input = torch.zeros(1, self.num_bands, self.patch_height, self.patch_width)
                output = cnn_module(dummy_input)
                cnn_module.train()
                return output.shape[1], output.shape[2], output.shape[3]  # C, H, W
            elif isinstance(cnn_module[0], nn.Conv1d):
                dummy_input = torch.zeros(1, self.num_bands, self.patch_height * self.patch_width)
                output = cnn_module(dummy_input)
                cnn_module.train()
                return output.shape[1], 1, output.shape[2]  # C, H=1 (dummy), W=SeqLen
            else:
                cnn_module.train()
                raise TypeError("First layer in cnn_module is not Conv1d or Conv2d")

    def forward(self, x):
        """
        Forward pass.
        Input x: (batch_size, patch_height, patch_width, num_bands)
        """
        # --- Input Permutation/Reshaping ---
        # PyTorch CNNs/Transformers usually expect Channels first
        # Target shape: (B, C, H, W) for Conv2D/Transformer patching
        # Target shape: (B, C, L) for Conv1D where L=H*W
        # # Target shape: (B, Features) for MLP
        # if self.model_type in ['cnn2d', 'transformer', 'cnn_transformer']:
        #      # (B, H, W, C) -> (B, C, H, W)
        #      x = x.permute(0, 3, 1, 2)
        if self.AE: x = x.view(x.size(0), self.encoder_C_deep, self.encoder_H, self.encoder_W)

        if self.model_type == 'cnn1d':
            # (B, H, W, C) -> (B, C, H*W)
            # x = x.permute(0, 3, 1, 2)
            x = x.reshape(x.size(0), self.num_bands, -1)
        elif self.model_type == 'mlp':
            # (B, H, W, C) -> (B, H*W*C)
            x = x.view(x.size(0), -1)

        # --- Model Body ---
        shared_output = None  # This will be the input to the regression head

        if self.model_type == 'cnn_transformer':
            # 1. CNN Backbone
            x = self.cnn_backbone(x)  # Output (B, C', H', W')
            # 2. Patch Embedding
            x = self.patch_embed(x)  # Output: (B, embed_dim, num_patches_h, num_patches_w)
            x = x.flatten(2)  # Output: (B, embed_dim, num_patches)
            x = x.transpose(1, 2)  # Output: (B, num_patches, embed_dim) - Transformer expects this
            # 3. Add CLS token and Positional Embedding
            if self.use_cls_token and self.use_pos_embed:
                batch_size = x.shape[0]
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
                x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)
                x = x + self.pos_embed  # Add pos embedding (broadcasting across batch)
            elif self.use_pos_embed:
                x = x + self.pos_embed[:, :self.num_patches, :]  # Add pos embedding
            # 4. Transformer Encoder
            x = self.transformer_encoder(x)  # Output: (B, num_patches+1 or num_patches, embed_dim)
            # 5. Aggregate output
            if self.use_cls_token:
                shared_output = x[:, 0]  # Select the CLS token's output
            else:
                shared_output = x.mean(dim=1)  # Mean pool across sequence dimension

        elif self.model_type == 'transformer':
            # 1. Patch Embedding (applied directly to input)
            x = self.patch_embed(x)  # Output: (B, embed_dim, num_patches_h, num_patches_w)
            x = x.flatten(2)  # Output: (B, embed_dim, num_patches)
            x = x.transpose(1, 2)  # Output: (B, num_patches, embed_dim)
            # 2. Add CLS token and Positional Embedding
            if self.use_cls_token and self.use_pos_embed:
                batch_size = x.shape[0]
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.pos_embed
            elif self.use_pos_embed:
                x = x + self.pos_embed[:, :self.num_patches, :]
            # 3. Transformer Encoder
            x = self.transformer_encoder(x)
            # 4. Aggregate output
            if self.use_cls_token:
                shared_output = x[:, 0]
            else:
                shared_output = x.mean(dim=1)

        elif self.model_type in ['mlp', 'cnn1d', 'cnn2d']:
            # Pass through shared MLP/CNN layers
            x = self.shared_layers(x)
            # Flatten output for regression head
            shared_output = x.view(x.size(0), -1)

        # --- Final Regression Heads ---
        regression_output1 = self.regression_head1(shared_output)
        if self.discriminator: return regression_output1
        if self.cls:
            cls_output = self.cls_head(shared_output)
            regression_output2 = self.regression_head2(shared_output)
            return cls_output, regression_output1, regression_output2
        if self.use_multi_head:
            # Pass through second regression head
            regression_output2 = self.regression_head2(shared_output)
            return regression_output1, regression_output2

        return regression_output1

class EncoderAE_3D(nn.Module):
    def __init__(self, config):  # input_size is tuple (H, W) or (D, H, W)
        super().__init__()
        self.config = config
        self.conv_dim = config.get('conv_dim', 2)
        self.use_encoder_fc = config.get('use_encoder_fc', True)
        self.is_variational = config.get('variational', False)  # Original VAE intention
        self.instanceNorm = config.get('encoder_instanceNorm', False)
        input_size = [8,8]
        encoder_conv_channels = config.get('encoder_channels', [128, 64, 32])[:]
        kernel_sizes = config.get('encoder_kernels', [3, 1, 1])[:len(encoder_conv_channels)]  # e.g., [K1, K2, K_deep]
        if self.conv_dim == 3: kernel_sizes.append(1)
        strides = config.get('encoder_strides', [1, 1, 1])[:len(encoder_conv_channels)]
        if self.conv_dim == 3: strides.append(1)
        D3_kernel = config.get('encoder_3D_kernel', 3)
        if self.conv_dim == 2: D3_kernel = None
        if self.conv_dim == 3: encoder_conv_channels.append(1)


        if not self.use_encoder_fc and self.is_variational:
            print("Warning: config.variational is True but use_encoder_fc is False. "
                  "Encoder will output feature maps, not mu/logvar for standard VAE. "
                  "Internally treating as non-variational for FC-related parts.")
            self.is_variational = False  # Override: No FC means no standard mu/logvar from FCs

        self.activation_fn = ACTIVATION_MAP[config.get('encoder_activation', 'relu')]()

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        if self.conv_dim == 3: input_channels = 1
        else: input_channels = 224
        current_channels = input_channels
        ConvNd = nn.Conv2d
        if not encoder_conv_channels:
            self.final_conv_out_channels = input_channels
        else:
            if self.conv_dim == 3: ConvNd = nn.Conv3d
            BatchNormNd = nn.BatchNorm2d if not self.instanceNorm else nn.InstanceNorm2d
            if self.conv_dim == 3:
                BatchNormNd = nn.BatchNorm3d
                if self.instanceNorm:
                    BatchNormNd = nn.InstanceNorm3d

            for i, out_channels in enumerate(encoder_conv_channels):
                kernel_size = [D3_kernel] if self.conv_dim == 3 else []
                kernel_size.extend([kernel_sizes[i],kernel_sizes[i]])
                padding = tuple(k // 2 for k in kernel_size)
                stride = [strides[i]] if self.conv_dim == 3 else []
                stride.extend([1,1])
                self.conv_layers.append(
                    ConvNd(in_channels=current_channels, out_channels=out_channels,
                           kernel_size=tuple(kernel_size), stride=tuple(stride), padding=tuple(padding))
                )
                self.bn_layers.append(BatchNormNd(out_channels))
                current_channels = out_channels
            self.final_conv_out_channels = current_channels

        with torch.no_grad():
            if self.conv_dim == 2:
                dummy_input_shape = (1, 224, input_size[0], input_size[1])
            else:  # conv_dim == 3
                dummy_input_shape = (1, 1, 224, input_size[0], input_size[1])

            dummy_input = torch.zeros(*dummy_input_shape)
            conv_output_for_flattening = dummy_input
            if self.conv_layers:
                for i in range(len(self.conv_layers)):
                    conv_output_for_flattening = self.conv_layers[i](conv_output_for_flattening)
                    conv_output_for_flattening = self.bn_layers[i](conv_output_for_flattening)
                    conv_output_for_flattening = self.activation_fn(conv_output_for_flattening)

            self.shape_before_flatten = conv_output_for_flattening.shape[1:]  # (C, H, W) or (C, D, H, W)
            self.flattened_size = conv_output_for_flattening.numel() // conv_output_for_flattening.shape[0]

        if self.use_encoder_fc:
            if 'latent_dim' not in config:
                raise ValueError("'latent_dim' must be in config if use_encoder_fc is True.")
            self.fc_mu = nn.Linear(self.flattened_size, self.flattened_size)
            if self.is_variational:  # Use the (potentially overridden) self.is_variational
                self.fc_logvar = nn.Linear(self.flattened_size, self.flattened_size)
            else:
                self.fc_logvar = None
        else:
            self.fc_mu = None
            self.fc_logvar = None

    def forward(self, x):
        x_conv_out = x
        if self.conv_dim == 3: x_conv_out = x_conv_out.unsqueeze(1)
        for i in range(len(self.conv_layers)):
            x_conv_out = self.conv_layers[i](x_conv_out)
            x_conv_out = self.bn_layers[i](x_conv_out)
            x_conv_out = self.activation_fn(x_conv_out)
        if self.conv_dim == 3: x_conv_out = x_conv_out.squeeze(1)
        if self.use_encoder_fc:
            x_flattened = x_conv_out.view(x_conv_out.size(0), -1)
            mu = self.fc_mu(x_flattened)
            if self.is_variational and self.fc_logvar is not None:
                logvar = self.fc_logvar(x_flattened)
                return mu, logvar, x_conv_out
            return mu, x_conv_out
        else:  # Not using FC layers
            return x_conv_out, x_conv_out

__all__ = ['ACTIVATION_MAP', 'SavGolFilterGPU', 'SpectralPredictorWithTransformer', 'EncoderAE_3D']
