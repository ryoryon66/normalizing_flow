import os
import tensorflow as tf
from PIL import Image
import io
import os
import argparse
import tensorflow as tf
from PIL import Image
import io


def parse_args():
    parser = argparse.ArgumentParser(description='Create GIF from TensorBoard')
    parser.add_argument('--event_file', type=str, help='Path to the event file')
    parser.add_argument('--tag', type=str, help='Tag of the target image')
    parser.add_argument('--image_dir', type=str, help='Directory to save the images', default='saved_gifs')
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.image_dir, exist_ok=True)
    # 画像を保存するリスト
    images = []
    # イベントファイルを読み込む
    dataset = tf.data.TFRecordDataset(args.event_file)
    for raw_record in dataset:
        event = tf.compat.v1.Event()
        event.ParseFromString(raw_record.numpy())
        for v in event.summary.value:
            # print(v.tag)
            if v.tag == args.tag:  # 画像のタグを確認
                # 画像データを取得
                img_str = v.image.encoded_image_string
                image = Image.open(io.BytesIO(img_str))
                images.append(image)
    # GIFとして保存
    gif_path = os.path.join(args.image_dir, f'{args.tag}.gif')
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)
    print(f'GIF saved at {gif_path}')

if __name__ == '__main__':
    main()


# os.makedirs(image_dir, exist_ok=True)

# # 画像を保存するリスト
# images = []

# # イベントファイルを読み込む
# dataset = tf.data.TFRecordDataset(event_file)
# for raw_record in dataset:
#     event = tf.compat.v1.Event()
#     event.ParseFromString(raw_record.numpy())
#     for v in event.summary.value:
#         # print(v.tag)
#         if v.tag == tag:  # 画像のタグを確認
#             # 画像データを取得
#             img_str = v.image.encoded_image_string
#             image = Image.open(io.BytesIO(img_str))
#             images.append(image)

# # GIFとして保存
# gif_path = os.path.join(image_dir, f'{tag}.gif')
# images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)
