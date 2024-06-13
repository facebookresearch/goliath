Datasheet in the format of "Datasheets for datasets" as described in

> Gebru, Timnit, et al. "Datasheets for datasets." Communications of the ACM 64.12 (2021): 86-92.

# Goliath-4 Dataset

<!-- TODO(julieta) add brief summary here, bibtex -->


## Motivation

1. **For what purpose was the dataset created?** *(Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.)*

    This dataset was created to facilitate the study of both (1) complete avatars -- that is, avatars that encompass the full body with high quality clothes and underlying body shape, and (2) generalizable avatars, or avatars that can be built from lower-quality, but more accessible captures.



1. **Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**

    The dataset was created by the Codec Avatars Team within Meta Reality Labs, at Meta.

1. **Who funded the creation of the dataset?** *(If there is an associated grant, please provide the name of the grantor and the grant name and number.)*

    Meta Platforms Inc.


1. **Any other comments?**

    None.





## Composition


1. **What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?** *(Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.)*

    Each instance comprises eight captures of the same person:
    1. a relightable capture of their head,
    2. a relightable capture of their hands,
    3. a full body capture with regular clothing,
    4. a full body capture with minimal clothing,

    As well as
    5. a mobile phone scan of their head,
    6. a mobile phone scan of thir hands,
    7. a mobile phone scan of their full body with regular clothing,
    8. a mobile phone scan of their full body with minimal clothing.

    All the phone captures are performed by the subjects themselves.



2. **How many instances are there in total (of each type, if appropriate)?**

    Our dataset constains 4 subjects, each containing all eight capture types described above.


3. **Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?** *(If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).)*

    The captures contain a wide range of poses of body motion, as well as face deformations and hand deformations for a particular subject, as well as samples of the changes of the appeareance of their heads and hands under different lighting conditions.

    The trade-off to this level of detail is that this is a small sample across subjects.


4. **What data does each instance consist of?** *(``Raw'' data (e.g., unprocessed text or images)or features? In either case, please provide a description.)*

    TODO(julieta)


5. **Is there a label or target associated with each instance? If so, please provide a description.**

    No.


6. **Is any information missing from individual instances?** *(If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.)*

    TODO(julieta)


7. **Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)?** *( If so, please describe how these relationships are made explicit.)*

    TODO(julieta)


8. **Are there recommended data splits (e.g., training, development/validation, testing)?** *(If so, please provide a description of these splits, explaining the rationale behind them.)*

    TODO(julieta)


9. **Are there any errors, sources of noise, or redundancies in the dataset?** *(If so, please provide a description.)*

    Assets are naturally noisy, since they are computed automatically. For example, segmentations might be imperfect, and kinematic tracking is can at times be inaccurate, same as 3d keypoints.
    Cameras are also naturally subject to multiple sources of noise such as thermal noise, chromatic aberrations, lense deformations, and manufacturing defects.


10. **Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** *(If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.)*

    The dataset is self-contained.


11. **Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals' non-public communications)?** *(If so, please provide a description.)*

    No.


12. **Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** *(If so, please describe why.)*

    No.


13. **Does the dataset relate to people?** *(If not, you may skip the remaining questions in this section.)*

    Yes, the dataset consists of captures of people.


14. **Does the dataset identify any subpopulations (e.g., by age, gender)?** *(If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.)*

    No.


15. **Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?** *(If so, please describe how.)*

    Yes, by looking at the provided images of the subjects.


16. **Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?** *(If so, please provide a description.)*

    No.

17. **Any other comments?**

    None.





## Collection Process


18. **How was the data associated with each instance acquired?** *(Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.)*

    The subjects were captured in a small high resolution dome with controllable lights, in a larger dome for full-body captures, and with a mobile phone for the phone captures.


19. **What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?** *(How were these mechanisms or procedures validated?)*

    For heads and hands, we used a high resolution dome with 172 cameras and a diameter of 2.4 m. This smaller dome also has lights that are fully turned on every 3rd frame, and are partially turned on in groups of 5 every 2 out of 3 frames.
    For full body captures we used a bigger high resolution dome with 220 cameras and a diameter of 5.5 m.

    The subjects were directed by a research assistant to express a range of emotions, make face, hand and full body deformations, and say phrases that comprise phonemes commonly occurring in English.

    All cameras take images at 30 fps, with a resolution of 4096 x 2668.


20. **If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?**

    In this dataset we aim to sample densely the appeareance of heads, hands, and full bodies for particular subjects.
    We achieve this sampling over appearance by prompting the subjects to follow pre-designed scripts designed to cover the range of motions and appearance changes in the human body.


21. **Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?**

    Subjects were at least 18 years old at the time of capture, provided their informed consent in writing, were compensated at a rate of USD $50 per hour, rounded to the next half hour.
    The captures are lead by a research assistant that prompted the subjects to make a range of motions with their heads, faces, and full bodies.
    A lot of custom software and hardware is involved in the process of data collection and asset generation. Please refer to the list of authors of our paper for a list of the people involved in this process.


22. **Over what timeframe was the data collected?** *(Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)?  If not, please describe the timeframe in which the data associated with the instances was created.)*

    The data was collected over 5 months, from March to July of 2023.


1. **Were any ethical review processes conducted (e.g., by an institutional review board)?** *(If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.)*

    We followed an internal research review process that includes, among other things, ethical considerations.
    At this moment we are not sharing these reviews publicly.


1. **Does the dataset relate to people?** *(If not, you may skip the remaining questions in this section.)*

    Yes; the dataset consists of captures of people.


1. **Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?**

    From the individuals directly.


1. **Were the individuals in question notified about the data collection?** *(If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.)*

    Yes; the subjects provided their informed written consent and were compensated for their time.


1. **Did the individuals in question consent to the collection and use of their data?** *(If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.)*

    Yes; the individuals gave their informed consent in writing for both capture and distribution of the data.
    We are not making the agreements public at this time.


1. **If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?** *(If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).)*

    The subjects were informed that they could drop out of the study at any time, and still be compensated for their time.
    At the moment we do not have a mechanism to revoke consent after the capture.


1. **Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?** *(If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.)*

    No.


1. **Any other comments?**

    None.





## Preprocessing/cleaning/labeling


1. **Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** *(If so, please provide a description. If not, you may skip the remainder of the questions in this section.)*

    To reduce the storage requirements of the dataset, we did the following:

    * Images were downsampled to 2048 x 1334 resolution.
    Compressing the RGB images with lossy avif compression at quality level 63.
    * Subsampling the frames temporal so that each hand, minimally clothed, regularly clothed and head capture has roughly 10,000 training frames, resulting in 5, 5, 10 and 10 frames per second respectively.


1. **Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?** *(If so, please provide a link or other access point to the "raw" data.)*

    Yes; the raw data is stored by the codec avatars team at Meta.
    The raw data is not publicly available, since it consists of several petabytes, and distributing that amount of data is infeasible.


1. **Is the software used to preprocess/clean/label the instances available?** *(If so, please provide a link or other access point.)*

    No; the methods used to produce the kinematic tracking, segmentation and keypoints are internal.


1. **Any other comments?**

    None.





## Uses


35. **Has the dataset been used for any tasks already?** *(If so, please provide a description.)*

    Yes, we provide code and checkpoints to build single-person
    * relightable head avatars based on relightable Gaussian splatting,
    * relightable hand avatars based on relightable volumetric primitives,
    * full body avatars with regular clothes, and
    * full-body avatars with minimal clothes, both based on dynamic meshes with neural textures.


36. **Is there a repository that links to any or all papers or systems that use the dataset?** *(If so, please provide a link or other access point.)*

    Not yet, as publications use this dataset we will update the github repository.


37. **What (other) tasks could the dataset be used for?**

    The head, hands, and body captures could be used to build full-body avatars with high resolution hands and heads.
    The corresponding paired mobile captures could be used to study the creation of high quality avatars from lower-quality captures.



38. **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?** *(For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks)  If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?)*

    None to our knowledge.


39. **Are there tasks for which the dataset should not be used?** *(If so, please provide a description.)*

    TODO


40. **Any other comments?**

    None.




## Distribution


1. **Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?** *(If so, please provide a description.)*

    Yes, the dataset is available under the CC-by-NC 4.0 License.


1. **How will the dataset be distributed (e.g., tarball  on website, API, GitHub)?** *(Does the dataset have a digital object identifier (DOI)?)*

    The dataset can be downloaded from AWS. A script to download it is available at https://github.com/facebookresearch/goliath/download.py.


1. **When will the dataset be distributed?**

    The dataset is available as of Wednesday June 12, 2024.


1. **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** *(If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.)*

    The dataset is licensed under a the CC-by-NC 4.0 license.


1. **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.)*

    Not to our knowledge.


1. **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.)*

    Not to our knowledge.


1. **Any other comments?**

    None.



## Maintenance


1. **Who is supporting/hosting/maintaining the dataset?**

    The open source team of the Codec Avatars Lab at Meta is supporting the dataset.
    The dataset is hosted on AWS S3 in a bucket paid for by Meta.


1. **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**

    You may contact Julieta Martinez via email at julietamartinez@meta.com.


1. **Is there an erratum?** *(If so, please provide a link or other access point.)*

    Currently, no. If we do encounter errors,  we will provide a list of issues on github as well as best practices to work around them.


1. **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances')?** *(If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?)*

    Same as previous.


1. **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?** *(If so, please describe these limits and explain how they will be enforced.)*

    No.


1. **Will older versions of the dataset continue to be supported/hosted/maintained?** *(If so, please describe how. If not, please describe how its obsolescence will be communicated to users.)*

    Yes; if we make changes, the data will be versioned.


1. **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** *(If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.)*

    We encourage other labs to collect and open source their high quality captures of human appeareance.
    While this is a costly effort, we will be happy to consider forming a super dataset of multiple captures taken by a diveristy of labs.


1. **Any other comments?**

    None.
