from simpleimage import SimpleImage
import numpy as np

DEFAULT_FILE = 'Stanford_256.jpg'


def main():  # {by: Charlie}
    # (1) load image–part of main
    image = SimpleImage(DEFAULT_FILE)

    # (2) get user’s choice of kernel
    choice = user_choice()

    # (3) get the kernel array
    kernel = create_kernel(choice)

    # (4) change image to grayscale
    make_grayscale(image)

    # (5) add 1px border to image
    image = add_border(image, 1)

    # (6) process the bordered image using the kernel
    final_image = process_image(image, kernel)

    # (7) show results
    final_image.show()


"""
    The user must choose a number between 1 and 6.  Each number is associated with a kernel as follows
    1 -> Gaussian blur
    2 -> Edge detection
    3 -> Line detection horizontal
    4 -> Line detection vertical
    5 -> Sobel edge detection horizontal
    6 -> Sobel edge detection vertical
"""


def user_choice():  # {by: Patricia}

    print("Please choose a kernel to use in image convolution.")
    print("")
    print("Here are the options to choose from:")
    choice_list = ["Gaussian blur", "Edge detection", "Line detection horizontal", "Line detection vertical", "Sobel edge detection horizontal", "Sobel edge detection vertical"]
    for i in range(len(choice_list)):
        print(str(i + 1) + " -> " + choice_list[i])
    choice = (input("Specify the number of the kernel you want to use: "))
    index = int(choice) - 1
    print("You chose " + choice_list[index])
    if choice == '1':
        kernel_choice = 'gaussian'
    elif choice == '2':
        kernel_choice = 'edge'
    elif choice == '3':
        kernel_choice = "lineh"
    elif choice == '4':
        kernel_choice = 'linev'
    elif choice == '5':
        kernel_choice = 'sobelh'
    elif choice == '6':
        kernel_choice = 'sobelv'

    return kernel_choice


'''
    return numpy array for kernel selected.
    choice: string.
'''
def create_kernel(choice):  # {by: Michael}

    kernel_dict ={}
    # put each kernel into dictionary
    kernel_dict['gaussian'] = [1/16, 1/8, 1/16, 1/8, 1/4, 1/8, 1/16, 1/8, 1/16]
    kernel_dict['edge'] = [-1, -1, -1, -1, 8, -1, -1, -1, -1]
    kernel_dict['lineh'] = [-1, -1, -1, 2, 2, 2, -1, -1, -1]
    kernel_dict['linev'] = [-1, 2, -1, -1, 2, -1, -1, 2, -1]
    kernel_dict['sobelh'] = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
    kernel_dict['sobelv'] = [-1, 0, 1, -2, 0, 2, -1, 0, 1]

    # get kernel list based on user's choice from above dictionary
    k_list = kernel_dict[choice]

    # convert k_list into numpy array and reshape to 9 x 1
    kernel = np.array(k_list)  # This is a 1 x 9 array
    kernel = np.reshape(kernel, (9, 1))  # now it's a 9 x 1 array!

    return kernel  # return the array


def make_grayscale(image):    # {by: Francesco}

    for pixel in image:
        pixel.red = (pixel.red + pixel.green + pixel.blue) // 3
        pixel.green = (pixel.red + pixel.green + pixel.blue) // 3
        pixel.blue = (pixel.red + pixel.green + pixel.blue) // 3
    return


def add_border(original_image, border_size):   # {by: Francesco}
    # average background intensity of entire image calculated for filling border pixels
    average_intensity = avg_px(original_image)
    # creating a new SimpleImage including border_size*2
    new_width = original_image.width + border_size * 2
    new_height = original_image.height + border_size * 2
    bordered_img = SimpleImage.blank(new_width, new_height)
    for y in range(bordered_img.height):
        for x in range(bordered_img.width):
            # setting border pixels
            if is_border(x, y, border_size, bordered_img):
                pixel = bordered_img.get_pixel(x, y)
                bordered_img.set_pixel(x, y, pixel)

                pixel.red = average_intensity
                pixel.green = average_intensity
                pixel.blue = average_intensity
            # setting other  pixels = gray_scale old image
            else:
                original_pixel = original_image.get_pixel(x - border_size, y - border_size)
                bordered_img.set_pixel(x, y, original_pixel)
    return bordered_img


def is_border(x, y, border_size, bordered_img):  # {by: Francesco}
    # top or  bottom border
    if y < border_size:
        return True
    if y >= bordered_img.height - border_size:
        return True
    # left or right border
    if x < border_size:
        return True
    if x >= bordered_img.width - border_size:
        return True

    return False


def avg_px(original_image):    # {by: Francesco}
    px_list = []
    for pixel in original_image:
        # input  average of every pixel RGB in list
        px_list.append((pixel.red + pixel.green + pixel.blue) // 3)
    # print(px_list)  #test for row above
        # calculating and returning average of all pixel average = average background intensity of entire image
    # print(sum(px_list) // len(px_list))  # test for row below = 131

    return sum(px_list)//len(px_list)


def process_image(img, kernel):  # {by: Anne}
    new_img = SimpleImage.blank(img.width, img.height)
    for y_ctr in range(1, img.height - 1):
        for x_ctr in range(1, img.width - 1):
            array = get_overlap(img, x_ctr, y_ctr)
            px = new_img.get_pixel(x_ctr, y_ctr)
            px.red = px.green = px.blue = np.dot(array, kernel)
    image = trim_crop(new_img, 1)
    return image


def trim_crop(original_img, trime_size):  # {by: Anne}
    new_width = original_img.width - 2 * trime_size
    new_height = original_img.height - 2 * trime_size
    new_img = SimpleImage.blank(new_width, new_height)
    for x in range(new_width):
        for y in range(new_height):
            old_x = x + trime_size
            old_y = y + trime_size
            orig_pixel = original_img.get_pixel(old_x, old_y)
            new_img.set_pixel(x, y, orig_pixel)
    return new_img


def get_overlap(image, x_ctr, y_ctr):  # {by: Charlie}
    """
    Purpose: for any pixel x, y in the bordered image, figure out the 9 pixels
    in the block centered at the anchor pixel. For clarity, the calling function
    will provide the correct anchor pixel location in the bordered image.
    return a 1x9 np.array
    :param image:
    :param x_ctr:
    :param y_ctr:
    :return array:
    """
    pixel_array = []
    for y in range(-1, 2, 1):
        for x in range(-1, 2, 1):
            pixel_array.append(image.get_pixel(x_ctr + x, y_ctr + y).red)
    return np.array(pixel_array)


if __name__ == '__main__':
    main()
