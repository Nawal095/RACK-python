# RACK: Automatic Query Reformulation for Code Search using Crowdsourced Knowledge

This is a Python Implementation for the Source code of RACK. The original Java implementation is [**here**](https://github.com/masud-technope/RACK-Server)

Please download the [**Word2Vec Model**](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) and unzip it in the root folder.

This implementation ships with a virtual environment. Please run the following command to activate the environment in Windows:

```
RACK-env\Scripts\activate
```

Below is an example of how to run the `RACK.py`. It will produce the results of *ranked_API_list*:

```
python RACK.py "How to parse HTML in Java?"
```

It will give the following output:

| API Token | KACScore | KPACScore | KKCScore | TotalScore |
|-----------|----------|-----------|---------|-------------|
| Class | 0.3684210526315789 | 0.0 | 0.0 | 0.0 |
| JSONObject | 0.4210526315789474 | 0.2222222222222222 | 0.0 | 0.0 |
| File | 0.7368421052631579 | 0.48148148148148145 | 1.0 | 0.0 |
| IOException | 0.7894736842105263 | 0.6296296296296295 | 1.0 | 0.0125 |
| Elements | 0.3684210526315789 | 0.5185185185185185 | 0.0 | 0.032499999999999994 |
| Node | 0 0 |.2222222222222222 | 0.0 | 0.05749999999999998 |
| Element | 0.5789473684210527 | 0.5555555555555555 | 0.118169226 | 0.0625 |
| Pattern | 0.3157894736842105 | 0.48148148148148145 | 0.118169226 | 0.07500000000000001 |
| Document | 1.0 | 1.0 | 0.118169226 | 0.08750000000000001 |
| JFrame | 0.2105263157894737 | 0.0 | 0.0 | 0.09750000000000002 |

We have also included a python notebook for the user convenience in terms of exploration.

### Citation

Please read the following papers for better understanding RACK.

-----------------------------------------
```
M. Masudur Rahman, Chanchal K. Roy and David Lo, "Automatic Query Reformulation for Code Search using 
Crowdsourced Knowledge", Journal of Empirical Software Engineering (EMSE), 56 pp.
```
**Download this paper:**  [<img src="https://img.icons8.com/color/344/pdf.png"
     alt="PDF" heigh="16px" width="16px" />](https://doi.org/10.1007/s10664-018-9671-0)
```
M. Masudur Rahman, Chanchal K. Roy and David Lo, "RACK: Automatic API Recommendation using Crowdsourced 
Knowledge", In Proceeding of The 23rd IEEE International Conference on Software Analysis, Evolution, and 
Reengineering (SANER 2016), pp. 349--359, Osaka, Japan, March 2016
```
**Download this paper:**  [<img src="https://img.icons8.com/color/344/pdf.png"
     alt="PDF" heigh="16px" width="16px" />](http://homepage.usask.ca/~masud.rahman/papers/masud-SANER2016.pdf)
```
M. Masudur Rahman, Chanchal K. Roy and David Lo, "RACK: Code Search in the IDE using Crowdsourced 
Knowledge", In Proceeding of The 39th International Conference on Software Engineering (ICSE 2017), 
pp. 51--54, Buenos Aires, Argentina, May, 2017
```
**Download this paper:**  [<img src="https://img.icons8.com/color/344/pdf.png"
     alt="PDF" heigh="16px" width="16px" />](http://homepage.usask.ca/~masud.rahman/papers/masud-ICSE2017.pdf)
     
You can cite the papers written by Rahman et al.
------------------------------------------------------------
```
@INPROCEEDINGS{emse2018masud,
author={Rahman, M. M. and Roy, C. K. and Lo, D.},
booktitle={EMSE}, 
title={Automatic Reformulation of Query for Code Search using Crowdsourced Knowledge},
year={2018},
pages={1--56} 
}
```
**Download this paper:**  [<img src="https://img.icons8.com/color/344/pdf.png"
     alt="PDF" heigh="16px" width="16px" />](https://doi.org/10.1007/s10664-018-9671-0)
```
@INPROCEEDINGS{saner2016masud,
author={Rahman, M. M. and Roy, C. K. and Lo, D.},
booktitle={Proc. SANER}, title={{RACK}: {A}utomatic {API} {R}ecommendation using {C}rowdsourced {K}nowledge},
year={2016},
pages={349--359} 
}
```
**Download this paper:**  [<img src="https://img.icons8.com/color/344/pdf.png"
     alt="PDF" heigh="16px" width="16px" />](http://homepage.usask.ca/~masud.rahman/papers/masud-SANER2016.pdf)
```
@INPROCEEDINGS{icse2017masud,
author={Rahman, M. M. and Roy, C. K. and Lo, D.},
booktitle={Proc. ICSE}, title={RACK: Code Search in the IDE using Crowdsourced Knowledge},
year={2017},
pages={51--54} 
}
```
**Download this paper:**  [<img src="https://img.icons8.com/color/344/pdf.png"
     alt="PDF" heigh="16px" width="16px" />](http://homepage.usask.ca/~masud.rahman/papers/masud-ICSE2017.pdf)
     
--------------------------------------------

Something not working as expected?
------------------------------------

Please contact **Noshin Nawal** (noshin.nawal@g.bracu.ac.bd) or [create a new issue](https://github.com/Nawal095/RACK-python/issues/new) for further information.
