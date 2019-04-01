import logger
import os


for root, dirs, files in os.walk("./runs"):
    for file in files:
        if file.endswith('.json') and 'archive' not in root:
            file_path = os.path.join(root, file)
            print('Loading: ', file_path)
            xp = logger.Experiment('dummy_name')
            xp.from_json(file_path)
            for opts in xp.visdom_win_opts.values():
                if 'legend' in opts:
                    opts.pop('legend')
            xp.to_visdom(visdom_opts={'server': 'http://localhost', 'port': 8098})
            print('Success!')
