import argparse
import numpy as np
import pandas as pd
import copy
import cv2


def open_window():
    cv2.namedWindow('', cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty('', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.setWindowProperty('', cv2.SCREE)
    cv2.startWindowThread()


def close_window():
    cv2.destroyAllWindows()


class HIC_Annotator(object):
    def __init__(self, excel_file, root_dir):
        self.excel_file = excel_file
        self.root_dir = root_dir
        self.curr_im = None
        self.curr_brown_im = None
        self.display = None
        self.curr_im_path = None
        self.space_on = False
        self.curr_sheet = None
        self.curr_label = None
        self.exit = False
        self.sheets = []
        self.curr_idx = None
        self.done = 0
        self.curr_sheet_idx = 0
        self.images = []
        self.labels = []
        self.num_imgs = 0
        self.data = {}
        color_dict = {
            1: np.array([1, 0, 0]),
            2: np.array([0.6, 0.2, 0.2]),
            3: np.array([0.33, 0.33, 0.33]),
            4: np.array([0.2, 0.2, 0.6]),
            5: np.array([0, 0, 1]),
            6: np.array([0.2, 1, 0.05]),
            7: np.array([71 / 255, 99 / 255, 1]),
            8: np.array([0.9, 1, 0.2]),
            9: np.array([1, 0.4, 1]),
            0: np.array([0, 0, 0])
        }
        self.color_dict = color_dict
        self.modes = ['PDL1', 'PD1', 'HE']
        self.curr_mode = 0

    def get_initial_image(self):
        i = 0
        for label in self.labels:
            if not pd.isna(label):  # skip over filed labels
                i += 1
                continue
            return i
        return i

    def run(self, debug=False):
        df = pd.read_excel(self.excel_file)

        self.data['HE'] = {'image path': list(df['HE_path']), 'label': [None] * len(list(df['HE_path']))}
        self.data['PDL1'] = {'image path': list(df['PDL1_path']), 'label': list(df['PDL1_label'])}
        self.data['PD1'] = {'image path': list(df['PD1_path']), 'label': list(df['PD1_label'])}

        df = self.data[self.modes[self.curr_mode]]
        self.images = df['image path']
        self.labels = df['label']
        num_imgs = len(self.images)
        self.num_imgs = num_imgs

        self.curr_idx = self.get_initial_image()
        while 0 <= self.curr_idx < len(self.images):
            im, label = self.images[self.curr_idx], self.labels[self.curr_idx]
            if self.exit:
                break
            try:
                if pd.isna(im):
                    self.curr_im_path = None
                    self.curr_im = np.zeros((1440, 2256, 3))
                    font = cv2.FONT_ITALIC
                    cv2.putText(self.curr_im, f'No Matching {self.modes[self.curr_mode]} Image',
                                (2256 // 6, 1440 // 2 - 100), font, 4, (0, 0, 1), 5,
                                cv2.LINE_AA)
                else:
                    self.curr_im_path = f'{self.root_dir}{im}'
                    print(self.curr_im_path)
                    self.curr_im = cv2.imread(self.curr_im_path)
                    if self.curr_im is None:
                        size = 1024
                        self.curr_im = np.zeros((size,size,3))
                        font = cv2.FONT_ITALIC
                        text = f'{self.curr_im_path} \n\n Image does not exists!'
                        textsize = cv2.getTextSize(text, font, 1, 2)[0]
                        # get coords based on boundary
                        textX = int(size//2 - (textsize[0] / 2))
                        textY = int(size//2 + (textsize[1] / 2))
                        # add text centered on image
                        cv2.putText(self.curr_im, text, (textX, textY), font, 1, (0, 0, 255), 2)

                    else:
                        self.curr_im = self.curr_im / 255

                font = cv2.FONT_ITALIC
                cv2.putText(self.curr_im, f'{self.modes[self.curr_mode]}', (35, 80), font, 2, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(self.curr_im, f'{self.curr_idx + 1}/{num_imgs}', (35, 125), font, 1, (0, 0, 0), 2,
                            cv2.LINE_AA)

                if not pd.isna(label):  # set label of current image
                    self.curr_label = label
                    self.curr_im = self.color_edge()
                else:
                    self.curr_label = None
                self.display = self.curr_im.copy()
                open_window()
                to_break = False
                while not to_break:
                    cv2.imshow('', self.display)
                    rv = cv2.waitKey()
                    if debug:
                        print(rv)
                    to_break = self.onkey(rv)

                if self.exit:
                    self.goodbye_message()
                    break
            except Exception as e:
                print(e)
                self.curr_im = None
                self.curr_im_path = None
                self.curr_label = None
                self.curr_idx += 1
                continue
        self.save_sheet()
        print('Done!')

    def goodbye_message(self):
        print('\n\n')
        pdl_left = len([l for l in self.data['PDL1']['label'] if pd.isna(l)])
        pd_left = len([l for l in self.data['PD1']['label'] if pd.isna(l)])
        print(f'{pdl_left} PDL1 images left to annotate')
        print(f'{pd_left} PDL images left to annotate')
        print('Saving results to file and exiting, please wait...')

    def save_sheet(self):
        df = pd.DataFrame({'HE_path': self.data['HE']['image path'],
                           'PDL1_path': self.data['PDL1']['image path'],
                           'PDL1_label': self.data['PDL1']['label'],
                           'PD1_path': self.data['PD1']['image path'],
                           'PD1_label': self.data['PD1']['label']})
        # df.to_csv(self.excel_file, index=False)
        df.to_excel(self.excel_file, index=False)

    def finish(self):
        close_window()
        self.curr_label = None
        self.exit = True

    def color_brown(self):
        self.curr_im = self.curr_im[..., ::-1]
        brown1 = np.array([0.41, 0.47, 0.6])
        brown2 = np.array([0.45, 0.4, 0.44])
        blue = np.array([0.39, 0.54, 0.8])

        # dist to brown:
        dist2brown1 = np.linalg.norm(self.curr_im.copy() - brown1, axis=-1, keepdims=True)
        dist2brown2 = np.linalg.norm(self.curr_im.copy() - brown2, axis=-1, keepdims=True)
        dist2brown = np.minimum(dist2brown1, dist2brown2)

        # dist blue brown:
        dist_blue_brown_1 = np.linalg.norm(brown1 - blue, axis=-1, keepdims=True)
        dist_blue_brown_2 = np.linalg.norm(brown2 - blue, axis=-1, keepdims=True)
        dist_blue_brown = np.minimum(dist_blue_brown_1, dist_blue_brown_2)

        eps = 0.03
        enhanced_im = np.concatenate([1 / (dist2brown + eps), np.ones_like(dist2brown) * (1 / (dist_blue_brown + eps)),
                                      eps * np.ones_like(dist2brown)], axis=2)
        enhanced_im = enhanced_im / enhanced_im.max()
        self.curr_im = self.curr_im[..., ::-1]
        return enhanced_im[..., ::-1]

    def color_edge(self, edge_width=15):
        im = self.curr_im.copy()
        if self.curr_label is not None:
            color = self.color_dict[self.curr_label]
            im[:edge_width, :] = color
            im[:, :edge_width, :] = color
            im[-edge_width:, :] = color
            im[:, -edge_width:, :] = color
        return im

    def onkey(self, event):
        """
        The main logic: performing the desired action according to the key pressed while analysing the current image
        :param event: The value return by cv2.waitKey() after pressing the keyboard:
        "escape" --> save results and exit
        "enter" --> save decision of current image and continue to next image
        "1" --> set label "positive" for current image
        "0" --> set label "negative" for current image
        "5" --> set label "None" for current image
        "space" --> toggle between HE, PDL1, PD.
        ">" --> discard decision for the current image and move to next image
        "<" --> discard decision for the current image and move to previous image
        "Tab" --> move to the first image in current sheet that is not annotated yet. if all are annotated - move the next sheet.
        :return: True if moving to next image or exiting, False if continue analysing the current image
        """
        if event == 27:  # escape
            self.finish()
            return True
        elif event == 13:  # enter
            if self.modes[self.curr_mode] != 'HE':
                self.labels[self.curr_idx] = copy.copy(self.curr_label)
                self.done += 1
                if self.done % 5 == 0:
                    print(f'auto saving progress...')
                    if self.done % 50 == 0:
                        pdl_left = len([l for l in self.data['PDL1']['label'] if pd.isna(l)])
                        pd_left = len([l for l in self.data['PD1']['label'] if pd.isna(l)])
                        print(f'{pdl_left} PDL1 images left to annotate')
                        print(f'{pd_left} PDL images left to annotate')
                    self.save_sheet()
            self.curr_idx += 1
            return True
        elif event == 49 and self.modes[self.curr_mode] != 'HE':  # 1
            self.curr_label = 1
            self.curr_im = self.color_edge()
            self.display = self.curr_im.copy()
            self.labels[self.curr_idx] = copy.copy(self.curr_label)
            cv2.imshow('', self.display)
        elif event == 50 and self.modes[self.curr_mode] != 'HE':  # 2
            self.curr_label = 2
            self.curr_im = self.color_edge()
            self.display = self.curr_im.copy()
            self.labels[self.curr_idx] = copy.copy(self.curr_label)
            cv2.imshow('', self.display)
        elif event == 51 and self.modes[self.curr_mode] != 'HE':  # 3
            self.curr_label = 3
            self.curr_im = self.color_edge()
            self.display = self.curr_im.copy()
            self.labels[self.curr_idx] = copy.copy(self.curr_label)
            cv2.imshow('', self.display)
        elif event == 52 and self.modes[self.curr_mode] != 'HE':  # 4
            self.curr_label = 4
            self.curr_im = self.color_edge()
            self.display = self.curr_im.copy()
            self.labels[self.curr_idx] = copy.copy(self.curr_label)
            cv2.imshow('', self.display)
        elif event == 53 and self.modes[self.curr_mode] != 'HE':  # 5
            self.curr_label = 5
            self.curr_im = self.color_edge()
            self.display = self.curr_im.copy()
            self.labels[self.curr_idx] = copy.copy(self.curr_label)
            cv2.imshow('', self.display)
        elif event == 116 and self.modes[self.curr_mode] != 'HE':  # t
            self.curr_label = 6
            self.curr_im = self.color_edge()
            self.display = self.curr_im.copy()
            self.labels[self.curr_idx] = copy.copy(self.curr_label)
            cv2.imshow('', self.display)
        elif event == 117 and self.modes[self.curr_mode] != 'HE':  # u
            self.curr_label = 7
            self.curr_im = self.color_edge()
            self.display = self.curr_im.copy()
            self.labels[self.curr_idx] = copy.copy(self.curr_label)
            cv2.imshow('', self.display)
        elif event == 99 and self.modes[self.curr_mode] != 'HE':  # c
            self.curr_label = 8
            self.curr_im = self.color_edge()
            self.display = self.curr_im.copy()
            self.labels[self.curr_idx] = copy.copy(self.curr_label)
            cv2.imshow('', self.display)
        elif event == 110 and self.modes[self.curr_mode] != 'HE':  # n
            self.curr_label = 9
            self.curr_im = self.color_edge()
            self.display = self.curr_im.copy()
            self.labels[self.curr_idx] = copy.copy(self.curr_label)
            cv2.imshow('', self.display)
        elif event == 48 and self.modes[self.curr_mode] != 'HE':  # 0
            self.curr_label = 0
            self.curr_im = self.color_edge()
            self.display = self.curr_im.copy()
            self.labels[self.curr_idx] = copy.copy(self.curr_label)
            cv2.imshow('', self.display)
        elif event == 32:  # space
            self.curr_mode = (self.curr_mode + 1) % 3
            df = self.data[self.modes[self.curr_mode]]
            self.images = df['image path']
            self.labels = df['label']
            return True
        elif event == 46:  # >
            self.curr_label = None
            self.curr_idx += 1
            return True
        elif event == 44:  # <
            self.curr_label = None
            self.curr_idx -= 1
            self.curr_idx = np.max([0, self.curr_idx])
            return True
        elif event == 9:  # Tab
            self.curr_label = None
            self.curr_idx = self.get_initial_image()
            return True
        elif event == 8:  # backspace
            self.curr_label = None
            self.labels[self.curr_idx] = copy.copy(self.curr_label)
            if self.curr_im_path is not None:
                self.curr_im = cv2.imread(self.curr_im_path) / 255
                font = cv2.FONT_ITALIC
                cv2.putText(self.curr_im, f'{self.modes[self.curr_mode]}', (35, 80), font, 2, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(self.curr_im, f'{self.curr_idx + 1}/{self.num_imgs}', (35, 125), font, 1, (0, 0, 0), 2,
                            cv2.LINE_AA)
                self.display = self.curr_im.copy()
                self.labels[self.curr_idx] = copy.copy(self.curr_label)
                cv2.imshow('', self.display)
            return False
        else:
            pass
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_file', type=str, help='full path to the excel file containing the '
                                                       'list of images to annotate.\n The excel can '
                                                       'contain multiple sheets, but each sheet should '
                                                       'contain 2 columns with the headers "image path" '
                                                       'and "label"', default='metadata/annotation_task.xlsx')
    parser.add_argument('--root_images_dir', type=str, help='The path to the root images dir', default='data')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    annotator = HIC_Annotator(args.excel_file, args.root_images_dir)
    annotator.run(debug=args.debug)
