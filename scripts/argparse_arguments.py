import argparse


def add_reconstruction_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="The grid resolution for standard reconstruction",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="The occupancy field threshold value",
    )
    parser.add_argument(
        "--mise_resolution",
        type=int,
        default=32,
        help="The initial resolution for MISE reconstruction",
    )
    parser.add_argument(
        "--upsampling_steps",
        type=int,
        default=3,
        help="The number of upsampling steps for the MISE reconstruction",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="The padding used for reconstructing the shape",
    )
    return parser


def add_ray_sampling_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--camera_distance",
        type=float,
        default=1.5,
        help="Distance from camera to the principal point. For our ShapeNet renderings this is set to 1.5",
    )
    parser.add_argument(
        "--near",
        type=float,
        default=0.5,
        help="Nearest distance for a ray",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=2.5,
        help="Farthest distance for a ray",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Image width",
    )
    parser.add_argument(
        "--num_point_samples",
        type=int,
        default=128,
        help="Number of sampled points along a ray",
    )
    parser.add_argument(
        "--up_vector",
        type=str,
        default="y",
        help="The vector pointing upwards. y for shapenet, z for the lego dataset",
    )
    return parser
