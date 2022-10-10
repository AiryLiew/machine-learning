import os
import numpy as np
import PIL.Image as pimage

def gen_data(bg_path,Uav_path,image_path,label_path):
    count = 0
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    with open(label_path, "w") as f:
        for filename in os.listdir(bg_path):
            background = pimage.open("{0}/{1}".format(bg_path, filename))
            img = background.convert("RGB")
            background_resize = img.resize((224, 224))
            #保存负样本
            background_resize.save("{}/{}.png".format(image_path, count))
            f.write("{}.png {} {} {} {} {}\n".format(count,0,0,0,0,0))
            #保证正负样本图片不重名
            count += 1
            name = np.random.randint(1, 34)
            Uav_img = pimage.open("{0}/{1}.png".format(Uav_path, name))

            new_w = np.random.randint(50, 100)
            new_h = np.random.randint(50, 100)
            resize_img = Uav_img.resize((new_w, new_h))
            rot_img = resize_img.rotate(np.random.randint(-45, 45))
            paste_x1 = np.random.randint(0, 224 - new_w)
            paste_y1 = np.random.randint(0, 224 - new_h)

            r, g, b, a = rot_img.split()
            background_resize.paste(rot_img, (paste_x1, paste_y1), mask=a)
            paste_x2 = paste_x1 + new_w
            paste_y2 = paste_y1 + new_h

            #保存正样本
            background_resize.save("{}/{}.png".format(image_path, count))
            f.write("{}.png {} {} {} {} {}\n".format(count,1,paste_x1,paste_y1,paste_x2,paste_y2))
            # 保证正负样本图片不重名
            count += 1
            if count >400:
                break

if __name__ == '__main__':
    bg_img  = r".\Dataset\Bg_Image"
    Uav_img  = r".\Dataset\Uav_Image"
    train_img = r".\Dataset\Train_image"
    validate_img  = r".\Dataset\Validate_image"
    test_img  = r".\Dataset\Test_image"
    train_label  = r"./Dataset/Train_label.txt"
    validate_label = r"./Dataset/Validate_label.txt"
    test_label = r"./Dataset/Test_label.txt"

    gen_data(bg_img ,Uav_img ,train_img,train_label)
    gen_data(bg_img ,Uav_img ,validate_img,validate_label)
    gen_data(bg_img ,Uav_img ,test_img,test_label)










