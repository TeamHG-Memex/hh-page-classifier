from setuptools import setup


setup(
    name='hh-page-clf',
    url='https://github.com/TeamHG-Memex/hh-page-classifier',
    packages=['hh_page_clf'],
    include_package_data=True,
    install_requires=[
        'attrs',
        'eli5',
        'html-text',
        'json_lines==0.2.0',
        'kafka-python==1.3.1',
        'numpy',
        'scikit-learn>=0.18<0.19',
        'scipy==0.18.1',
        'tldextract',
        'tqdm',
        'ujson',
    ],
    entry_points={
        'console_scripts': [
            'hh-page-clf-service=hh_page_clf.service:main',
            'hh-page-clf-train=hh_page_clf.train:main',
            'train-lda=hh_page_clf.train_lda:main',
            'extract-texts=hh_page_clf.train_lda:extract_texts',
            'dmoz-to-csv=hh_page_clf.dmoz_reader:to_csv',
        ],
    },
    classifiers=[
        'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
)
