from setuptools import setup


setup(
    name='hh-page-clf',
    url='https://github.com/TeamHG-Memex/hh-page-classifier',
    packages=['hh_page_clf'],
    include_package_data=True,
    install_requires=[
        'eli5>=0.0.7',
        'html-text',
        'kafka-python==1.3.1',
        'numpy',
        'scikit-learn==0.18',
        'scipy==0.18.1',
        'tldextract',
    ],
    entry_points = {
        'console_scripts': [
            'hh-page-clf-service=hh_page_clf.service:main',
            'hh-page-clf-train=hh_page_clf.train:main',
        ],
    },
    classifiers=[
        'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
)
