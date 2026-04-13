import torch
import shutil
import os
import logging
import torchvision.utils as tvu


logger = logging.getLogger(__name__)


def setup_logger(log_dir, name='lightendiffusion'):
    """Setup file + console logger with proper formatting.

    Creates a logger that outputs to both console (INFO level) and
    a log file (DEBUG level) in the specified directory.

    Args:
        log_dir: Directory to save the log file.
        name: Logger name. Default: 'lightendiffusion'.

    Returns:
        logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called multiple times
    if log.handlers:
        return log

    # Console handler (INFO and above)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(ch_fmt)
    log.addHandler(ch)

    # File handler (DEBUG and above)
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setLevel(logging.DEBUG)
    fh_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fh_fmt)
    log.addHandler(fh)

    return log


def log_config(config, log=None):
    """Pretty-print the config namespace to the logger and console.

    Args:
        config: argparse.Namespace config object.
        log: Optional logger. If None, prints to stdout.
    """
    def _format_namespace(ns, indent=0):
        lines = []
        prefix = '  ' * indent
        for key, value in sorted(vars(ns).items()):
            if hasattr(value, '__dict__') and not callable(value):
                lines.append(f'{prefix}{key}:')
                lines.extend(_format_namespace(value, indent + 1))
            else:
                lines.append(f'{prefix}{key}: {value}')
        return lines

    header = '=' * 50 + '\n  CONFIGURATION\n' + '=' * 50
    config_str = '\n'.join(_format_namespace(config))
    full = f'\n{header}\n{config_str}\n{"=" * 50}'

    if log:
        log.info(full)
    else:
        print(full)


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)


def save_checkpoint(state, filename, is_best=False):
    """Save checkpoint with optional best-model copy.

    Args:
        state: Dictionary with model state, optimizer, etc.
        filename: Base filename (without extension).
        is_best: If True, also saves a copy as 'model_best.pth.tar'.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    filepath = filename + '.pth.tar'
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(os.path.dirname(filename), 'model_best.pth.tar')
        shutil.copyfile(filepath, best_path)


def load_checkpoint(path, device):
    if device is None:
        return torch.load(path, map_location='cpu')
    else:
        return torch.load(path, map_location=device)
