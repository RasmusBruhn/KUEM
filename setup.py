from distutils.core import setup
setup(
  name = 'KUEM',         # How you named your package folder (MyLib)
  packages = ['KUEM'],   # Chose the same as "name"
  version = '1.6',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A sim class to setup, simulate and create videos and plots of electrostatics and dynamics systems',   # Give a short description about your library
  author = 'Rasmus Bruhn Nielsen',                   # Type in your name
  author_email = 'rasmusbruhnnielsen@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/RasmusBruhn/KUEM',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/RasmusBruhn/KUEM/archive/refs/tags/v1.6.tar.gz',    # I explain this later on
  keywords = ['Electromagnetism', 'Simulation', 'University of Copenhagen'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'scipy',
          'matplotlib',
          'opencv-python',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.8',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
)
