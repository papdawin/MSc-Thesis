## Többrétegű kereskedelmi és kollaborációs hálózatok jellemzőinek idősoros vizsgálata, anomáliák, sokkok detektálása
#### Adattudomány MSc Diplomadolgozat
###### Veszprém, 2025.05.03

[![en](https://img.shields.io/badge/version-English-blue.svg)](https://github.com/papdawin/MSc-Thesis/blob/master/README.md)
[![hu](https://img.shields.io/badge/version-Hungarian-brown.svg)](https://github.com/papdawin/MSc-Thesis/blob/master/README.hu.md)

Diplomadolgozatomban elsősorban a globális kereskedelmi hálózatok elemzésére
fókuszáltam, különös tekintettel a központisági mutatókra, autoencoder modellekre, gráf
neurális hálózatokra (GNN), embedding technikákra majd ezek segítésével az idősoros
változások elemzésére és anomália-detektálásra.
Az elemzés alapját az OECD Inter-Country Input-Output (ICIO) adatbázis képezte,
amely 1995 és 2020 között 76 ország kereskedelmi adatait tartalmazza, szektorokra
bontva az ISIC Rev. 4 szabvány szerint. Ezeket az adatokat először adjacencia mátrixból
él lista formátumba alakítottam, hogy aggregált és többrétegű hálózatok hozhassak létre.
Az elkészült hálózatokból NetworkX gráfokat készítettem, majd a Gráf-neurális
megoldások esetén PyGData objektummá alakítottam.
A készített gráfokon megvizsgáltam több központisági mutatót, az így kapott
eredményeket elemeztem, majd vizualizáltam és a kapott eredményeket dokumentáltam.
Az adatok idősorelemzéséhez egy autoencoder alapú modellt és egy beágyazó modellt is
készítettem. Az autoencoder alapú megoldás lehetővé tette az anomália-detektálást,
vagyis a szokatlan vagy kiugró kereskedelmi mintázatok azonosítását a hálózatban amik
gazdasági sokkokra utalhatnak.
Az embedding alapú megoldás segítségével egy idősoros vektorreprezentációt képeztem,
amivel az évek során történt kereskedelmi hálózati átalakulásokat modelleztem. Majd
ezeket a megközelítéseket a Gráf Neurális Hálózat segítségével is implementáltam, az így
képzett GNN beágyazás a gráfok komplex kapcsolati struktúrájának tanulására
használtam, amelyek képesek voltak a csomópontok és élek közötti nemlineáris
összefüggések feltárására, így tovább javítva az anomáliák felismerését és az
adathalmazban lévő hálózati dinamikák megértését.
Összességében a kutatás során központisági mutatókat elemeztem, beégyazás
segítségével idősori változásokat, autoencoderrel pedig anomáliákat detektáltam a
kereskedelmi hálózatban. Majd ezekre készítettem egy Gráf Neurális alternatívát, amit
kiértékeltem és összehasonlítottam a hagyományos módszerrel. A kapott eredmények
lehetővé tették a hálózati dinamikák mélyebb megértését és a gazdasági rendellenességek
pontosabb azonosítását.
