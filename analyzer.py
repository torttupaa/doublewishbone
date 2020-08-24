import math
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL.shaders
import numpy as np
import pyrr
import tinyik
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import tkinter as tk

class Model3Dvbo():
    def __init__(self, filename, index, img_index, text_list, text_koko, nosto):

        self.filename = filename
        self.text_koko = text_koko
        self.img_index = img_index
        self.nosto = nosto
        self.verticies = []
        self.normals = []
        self.index = index  # indexi CallListia varten Ja TEX iideetä
        self.tex_koords = []
        self.pinta_kaikilla = []
        self.alustus()

        #print(self.filename)
        #print("poly ",len(self.pinta_kaikilla)/3," vert ", len(self.verticies),"\n")

        self.lattia_nosto(nosto)
        self.tex_koords_kerrottu = self.tex_koord_kerroin()
        self.lataa_text(text_list)

        self.vbo = self.vboize()
        #self.VerteXnormal()

        self.tangenttilista = self.tangenttitehdas()
        self.vbo = self.tangentti_mukaan_vbohon()
        self.vbo = np.array(self.vbo, dtype=np.float32)

        self.VBOVAOgenerointi()



    def alustus(self):
        self.file = open(self.filename, "r")

        for rivi in self.file:
            pygame.event.get()
            rivi_lista = rivi.split()
            try:
                tyyppi = rivi_lista[0]
                data = rivi_lista[1:]

                if tyyppi == "v":
                    x, y, z = data
                    vertex = (float(x), float(y), float(z))
                    self.verticies.append(vertex)

                elif tyyppi == "vt":
                    x = data[0]
                    y = data[1]

                    tex_coords = (float(x), float(y))
                    self.tex_koords.append(tex_coords)

                elif tyyppi == "vn":
                    x, y, z = data
                    normal = (float(x), float(y), float(z))
                    self.normals.append(normal)


                elif tyyppi == "f":
                    for v_vt_vn in data:
                        vtn = list((v_vt_vn.split("/")))
                        self.pinta_kaikilla.append(vtn)

            except:
                ValueError

        self.file.close()
    def vboize(self):
        VBO_alku = []
        VBOO = []
        self.vertexit_jarjestyksessa = []
        self.UV_jarjestyksessa = []
        self.normaalit_jarjestyksessa = []
        for alkio in self.pinta_kaikilla:
            pygame.event.get()
            v = (self.verticies[int(alkio[0]) - 1])
            vt = (self.tex_koords_kerrottu[int(alkio[1]) - 1])
            vn = (self.normals[int(alkio[2]) - 1])

            self.vertexit_jarjestyksessa.append(v)
            self.UV_jarjestyksessa.append(vt)
            self.normaalit_jarjestyksessa.append(vn)

            VBO_alku.append(v)
            VBO_alku.append(vt)
            VBO_alku.append(vn)
        for osa in VBO_alku:
            pygame.event.get()
            for OSA in osa:
                VBOO.append(OSA)
        del self.normals
        del self.tex_koords
        del self.tex_koords_kerrottu
        return VBOO
    def tangenttitehdas(self):
        #tangenttilista_palanen = []
        tangenttilista = []
        for x in range(len(self.vertexit_jarjestyksessa)):
            pygame.event.get()
            if x % 3 == 0:
                # edge1
                v1_v00 = self.vertexit_jarjestyksessa[x + 1][0] - self.vertexit_jarjestyksessa[x][0]
                v1_v01 = self.vertexit_jarjestyksessa[x + 1][1] - self.vertexit_jarjestyksessa[x][1]
                v1_v02 = self.vertexit_jarjestyksessa[x + 1][2] - self.vertexit_jarjestyksessa[x][2]

                # edge2
                v2_v00 = self.vertexit_jarjestyksessa[x + 2][0] - self.vertexit_jarjestyksessa[x][0]
                v2_v01 = self.vertexit_jarjestyksessa[x + 2][1] - self.vertexit_jarjestyksessa[x][1]
                v2_v02 = self.vertexit_jarjestyksessa[x + 2][2] - self.vertexit_jarjestyksessa[x][2]

                # deltaUV1
                uv1_uv00 = self.UV_jarjestyksessa[x + 1][0] - self.UV_jarjestyksessa[x][0]
                uv1_uv01 = self.UV_jarjestyksessa[x + 1][1] - self.UV_jarjestyksessa[x][1]

                # deltaUV2
                uv2_uv00 = self.UV_jarjestyksessa[x + 2][0] - self.UV_jarjestyksessa[x][0]
                uv2_uv01 = self.UV_jarjestyksessa[x + 2][1] - self.UV_jarjestyksessa[x][1]

                try:
                    r = 1 / ((uv1_uv00 * uv2_uv01) - (uv1_uv01 * uv2_uv00))
                except:
                    r = 1
                    print("fail")

                Tx = ((v1_v00 * uv2_uv01) - (v2_v00 * uv1_uv01)) * r
                Ty = ((v1_v01 * uv2_uv01) - (v2_v01 * uv1_uv01)) * r
                Tz = ((v1_v02 * uv2_uv01) - (v2_v02 * uv1_uv01)) * r

                Tangentti = [Tx, Ty, Tz]
                tangenttilista.append(Tangentti)
                tangenttilista.append(Tangentti)
                tangenttilista.append(Tangentti)
        del self.UV_jarjestyksessa
        del self.vertexit_jarjestyksessa
        return tangenttilista
    def tangentti_mukaan_vbohon(self):
        prosessoitu_vbo = []
        y = -1
        for x in range(len(self.vbo)):
            pygame.event.get()
            if x % 8 == 0:
                prosessoitu_vbo.append(self.tangenttilista[y][0])
                prosessoitu_vbo.append(self.tangenttilista[y][1])
                prosessoitu_vbo.append(self.tangenttilista[y][2])
                y += 1
            prosessoitu_vbo.append(self.vbo[x])
        del prosessoitu_vbo[0]
        del prosessoitu_vbo[0]
        del prosessoitu_vbo[0]
        prosessoitu_vbo.append(self.tangenttilista[-1][0])
        prosessoitu_vbo.append(self.tangenttilista[-1][1])
        prosessoitu_vbo.append(self.tangenttilista[-1][2])
        del self.tangenttilista
        return prosessoitu_vbo
    def tex_koord_kerroin(self):
        kerrotut = []
        for tex in self.tex_koords:
            pygame.event.get()
            kerrotut.append(((tex[0] / self.text_koko), (tex[1] / self.text_koko)))
        return kerrotut
    def lataa_text(self, text_list):
        if text_list[0] == 2:
            reflect_plane_texture = glGenTextures(1, self.img_index)
            glBindTexture(GL_TEXTURE_2D, reflect_plane_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1280, 800, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glBindTexture(GL_TEXTURE_2D, 0)
        else:
            y=0
            for text in text_list:
                image = pygame.image.load(text)
                width = image.get_width()
                height = image.get_height()
                image = pygame.image.tostring(image, "RGBA", True)

                texture_ID = self.img_index + y
                y += 1
                ID = glGenTextures(1, texture_ID)
                glBindTexture(GL_TEXTURE_2D, ID)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
                glGenerateMipmap(GL_TEXTURE_2D)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    def VBOVAOgenerointi(self):

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, (len(self.vbo) * 4), self.vbo, GL_STATIC_DRAW)

        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)

        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(32))
        glEnableVertexAttribArray(3)


        glBindVertexArray(0)
    def instanceBuffer(self, instanssi_mat_list, instanssi_trans_list):

        self.instance_array = np.array(instanssi_mat_list, dtype=np.float32)
        self.instance_array2 = np.array(instanssi_trans_list, dtype=np.float32)


        instanceM_VBO = glGenBuffers(1, self.index)
        glBindBuffer(GL_ARRAY_BUFFER, instanceM_VBO)
        glBufferData(GL_ARRAY_BUFFER, 64 * len(self.instance_array), self.instance_array, GL_STATIC_DRAW)

        glBindVertexArray(self.index)

        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(0))
        glEnableVertexAttribArray(4)

        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(16))
        glEnableVertexAttribArray(5)

        glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(32))
        glEnableVertexAttribArray(6)

        glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(48))
        glEnableVertexAttribArray(7)

        glVertexAttribDivisor(4, 1)
        glVertexAttribDivisor(5, 1)
        glVertexAttribDivisor(6, 1)
        glVertexAttribDivisor(7, 1)

        instanceT_VBO = glGenBuffers(1, self.index)
        glBindBuffer(GL_ARRAY_BUFFER, instanceT_VBO)
        glBufferData(GL_ARRAY_BUFFER, 64 * len(self.instance_array2), self.instance_array2, GL_STATIC_DRAW)

        glBindVertexArray(self.index)

        glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(0))
        glEnableVertexAttribArray(8)

        glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(16))
        glEnableVertexAttribArray(9)

        glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(32))
        glEnableVertexAttribArray(10)

        glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(48))
        glEnableVertexAttribArray(11)

        glVertexAttribDivisor(8, 1)
        glVertexAttribDivisor(9, 1)
        glVertexAttribDivisor(10, 1)
        glVertexAttribDivisor(11, 1)

        randomlista = []
        for z in range(len(self.instance_array)):
            randomlista.append(np.random())

        randomlista = np.array(randomlista, dtype=np.float32)

        random_VBO = glGenBuffers(1, self.index)
        glBindBuffer(GL_ARRAY_BUFFER, random_VBO)
        glBufferData(GL_ARRAY_BUFFER, 4 * len(randomlista), randomlista, GL_STATIC_DRAW)

        glBindVertexArray(self.index)

        glVertexAttribPointer(12, 1, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(12)
        glVertexAttribDivisor(12, 1)

    def piirra(self, shader, shader_type):
        if shader_type == "no_light_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_index)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
        elif shader_type == "Ishader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_index)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 1)
            tex = glGetUniformLocation(shader, "normalMap")
            glUniform1i(tex, 1)

            glBindVertexArray(self.index)
            glDrawArraysInstanced(GL_TRIANGLES, 0, int(len(self.vbo) / 11), len(self.instance_array))
            glBindVertexArray(0)
        elif shader_type == "perus_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_index)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 1)
            tex = glGetUniformLocation(shader, "normalMap")
            glUniform1i(tex, 1)
            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)

        elif shader_type == "T_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 8)
            tex = glGetUniformLocation(shader, "blendMap")
            glUniform1i(tex, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.img_index)
            tex = glGetUniformLocation(shader, "BGTex")
            glUniform1i(tex, 1)

            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 2)
            tex = glGetUniformLocation(shader, "rTex")
            glUniform1i(tex, 2)

            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 4)
            tex = glGetUniformLocation(shader, "gTex")
            glUniform1i(tex, 3)

            glActiveTexture(GL_TEXTURE4)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 6)
            tex = glGetUniformLocation(shader, "bTex")
            glUniform1i(tex, 4)


            glActiveTexture(GL_TEXTURE5)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 1)
            tex = glGetUniformLocation(shader, "BGnormalMap")
            glUniform1i(tex, 5)

            glActiveTexture(GL_TEXTURE6)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 3)
            tex = glGetUniformLocation(shader, "RnormalMap")
            glUniform1i(tex, 6)

            glActiveTexture(GL_TEXTURE7)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 5)
            tex = glGetUniformLocation(shader, "GnormalMap")
            glUniform1i(tex, 7)

            glActiveTexture(GL_TEXTURE8)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 7)
            tex = glGetUniformLocation(shader, "BnormalMap")
            glUniform1i(tex, 8)

            glBindVertexArray(self.index)
            #glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
        elif shader_type == "T2_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_index)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 1)
            tex = glGetUniformLocation(shader, "normalMap")
            glUniform1i(tex, 1)

            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)

        elif shader_type == "F_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, 44)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, 46)
            tex = glGetUniformLocation(shader, "toinen")
            glUniform1i(tex, 1)

            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, 48)
            tex = glGetUniformLocation(shader, "prev1")
            glUniform1i(tex, 2)

            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)

    def lattia_nosto(self, nosto):
        noni = []
        for vert in self.verticies:
            pygame.event.get()
            asd = vert[1] + nosto
            uudet = [vert[0], asd, vert[2]]
            noni.append(uudet)
            uudet = []
        self.verticies = noni
    def VerteXnormal(self):
        self.vertexnormal = []
        for gynther in range(len(self.vertexit_jarjestyksessa)):
            self.vertexnormal.append(self.vertexit_jarjestyksessa[gynther])
            self.vertexnormal.append(self.normaalit_jarjestyksessa[gynther])
        del self.normaalit_jarjestyksessa
class ASETUSARVOJA():
    def __init__(self):
        self.P = 0
        self.V = 0
        self.Vlast = pyrr.matrix44.create_from_translation(pyrr.Vector3([0,0,0]))
        self.Lpos = 0
        self.Vpos = 0
        self.Vref = 0

        self.zoom = 0

        self.display = (1280, 800)

        self.hahmo_koords = [-1.334119896369329, -1.15, -3.998029534518906]

        self.Max_nopeus = 7
        self.Kiihtyvyys = 0.01  # ei vaikuta kiihtyvyyteen mitenkaan vaan nopeuteen :D

        self.XZ_move_ws = 0
        self.XZ_MOVE_ws = 0
        self.XZ_move_vektori_ws = [0.0, 0.0]

        self.XZ_move_ad = 0
        self.XZ_MOVE_ad = 0
        self.XZ_move_vektori_ad = [0.0, 0.0]

        self.sens = 0.3
        self.x_rotation = 0#62831.853
        self.y_rotation = 0#62831.853
        self.mousepos_edellinen_x = 640
        self.mousepos_edellinen_y = 400
        self.invert_y = True

        self.right = 0
        self.rotator = 300000
        self.updown = 0

        self.cube_pos = self.objposMatrix([0.0, 0.0, 0.0])

        self.AL = 0.5
        self.SL = 1.0

        self.alatuki_etu = np.array([0, 0, 0])
        self.alatuki_taka = np.array([0, 0.2, 1])
        self.alatuki_kakseli_init = np.array([0.6, -0.1, 0.6])
        self.alatuki_kakseli = []

        self.ylatuki_etu = np.array([0.2, 0.5, 0.2])
        self.ylatuki_taka = np.array([0.2, 0.4, 0.85])
        self.ylatuki_kakseli_init = np.array([0.5, 0.6, 0.7])

        self.euler_index = 0
        self.ala_etu_euler = []
        self.ala_taka_euler = []
        self.yla_etu_euler = []
        self.yla_taka_euler = []
        self.valipala_euler = []

        self.alatuki_etu_len = 0
        self.alatuki_taka_len = 0
        self.ylatuki_etu_len = 0
        self.ylatuki_taka_len = 0
        self.valipala_len = 0

        self.jousto = []
        self.rengastie = []
        self.rengas_keski = []

        self.PLOT = False



    def objposMatrix(self, pos):
        objpos = pyrr.matrix44.create_from_translation(pyrr.Vector3(pos))
        return objpos

###render
def render(asetusarvo, tube, shader, shader_type):

    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")
    glUniform1f(glGetUniformLocation(shader, "specularStrenght"), 2.0)
    glUniform1f(glGetUniformLocation(shader, "ambientSTR"), asetusarvo.AL)
    glUniform1f(glGetUniformLocation(shader, "sunlightSTR"), asetusarvo.SL)

    # alatukietu
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3(asetusarvo.alatuki_etu+np.array([0,-asetusarvo.rengastie[asetusarvo.euler_index][1],0])))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    #rot_xyz = [asetusarvo.x_rotation, asetusarvo.y_rotation, 0]
    #rot_xyz = [-299.52501626,   -8.79177666,  158.24330663]
    rot_xyz = asetusarvo.ala_etu_euler[asetusarvo.euler_index] #INDEXILLA
    rot = rotate_XYZ(rot_xyz)
    tubelen = asetusarvo.alatuki_etu_len
    scale = tubelenght(tubelen)
    #obj_scale = 0.1
    #scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    tube.piirra(shader, shader_type)

    # alatukitaka
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3(asetusarvo.alatuki_taka + np.array([0, -asetusarvo.rengastie[asetusarvo.euler_index][1], 0])))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    #rot_xyz = [asetusarvo.x_rotation, asetusarvo.y_rotation, 0]
    rot_xyz = asetusarvo.ala_taka_euler[asetusarvo.euler_index]  # INDEXILLA
    #rot_xyz = [0, 0, 0]
    rot = rotate_XYZ(rot_xyz)
    tubelen = asetusarvo.alatuki_taka_len
    scale = tubelenght(tubelen)
    #obj_scale = 0.1
    #scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    tube.piirra(shader, shader_type)

    # valipala
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3(asetusarvo.alatuki_kakseli[asetusarvo.euler_index]+np.array([0,-asetusarvo.rengastie[asetusarvo.euler_index][1],0])))  #INDEXILLA
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    #rot_xyz = [asetusarvo.x_rotation, asetusarvo.y_rotation, 0]
    rot_xyz = np.rad2deg(-asetusarvo.valipala_euler[asetusarvo.euler_index])+np.array([0,0,90]) #INDEXILLA
    rot = rotate_XYZ(rot_xyz)
    tubelen = asetusarvo.valipala_len
    scale = tubelenght(tubelen)
    #obj_scale = 0.1
    #scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    tube.piirra(shader, shader_type)

    # ylatukietu
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3(asetusarvo.ylatuki_etu + np.array([0, -asetusarvo.rengastie[asetusarvo.euler_index][1], 0])))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    # rot_xyz = [asetusarvo.x_rotation, asetusarvo.y_rotation, 0]
    rot_xyz = asetusarvo.yla_etu_euler[asetusarvo.euler_index]  # INDEXILLA
    #rot_xyz = [0, 0, 0]
    rot = rotate_XYZ(rot_xyz)
    tubelen = asetusarvo.ylatuki_etu_len
    scale = tubelenght(tubelen)
    # obj_scale = 0.1
    # scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    tube.piirra(shader, shader_type)

    # ylatukitaka
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3(asetusarvo.ylatuki_taka + np.array([0, -asetusarvo.rengastie[asetusarvo.euler_index][1], 0])))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    # rot_xyz = [asetusarvo.x_rotation, asetusarvo.y_rotation, 0]
    rot_xyz = asetusarvo.yla_taka_euler[asetusarvo.euler_index]  # INDEXILLA
    #rot_xyz = [0,0,0]
    rot = rotate_XYZ(rot_xyz)
    tubelen = asetusarvo.ylatuki_taka_len
    scale = tubelenght(tubelen)
    # obj_scale = 0.1
    # scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    tube.piirra(shader, shader_type)

    #rengas
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3(asetusarvo.rengas_keski[asetusarvo.euler_index] + np.array([0, -asetusarvo.rengastie[asetusarvo.euler_index][1], 0])))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = np.rad2deg(np.array([-asetusarvo.valipala_euler[asetusarvo.euler_index][0],0,-asetusarvo.valipala_euler[asetusarvo.euler_index][2]]))# + np.array([0, 0, 90])  # INDEXILLA
    #rot_xyz = [0,0,0]
    rot = rotate_XYZ(rot_xyz)
    #tubelen = 0.05
    #scale = tubelenght(tubelen)
    D = 0.6
    leveys=0.15
    scale = rengass(leveys,D)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    tube.piirra(shader, shader_type)

def render_nolight(asetusarvo, shader, shader_type, skybox):

    glUseProgram(shader)

    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.V)
    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")

    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0,0,0]))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [0, 0, 0]#[0, pygame.time.get_ticks() / 500,0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = 30
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    skybox.piirra(shader,shader_type)
    glUseProgram(0)

def render_terrain(lattia,asetusarvo,shader,shader_type):
    # lattia
    glUseProgram(shader)

    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.V)
    glUniformMatrix4fv(glGetUniformLocation(shader, "viewPos"), 1, GL_FALSE, asetusarvo.Vpos)
    glUniformMatrix4fv(glGetUniformLocation(shader, "lightpos"), 1, GL_FALSE, asetusarvo.Lpos)
    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")
    glUniform1f(glGetUniformLocation(shader, "specularStrenght"), 0.1)
    glUniform1f(glGetUniformLocation(shader, "ambientSTR"), asetusarvo.AL)
    glUniform1f(glGetUniformLocation(shader, "sunlightSTR"), asetusarvo.SL)

    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0,0,0]))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [179, 0, 0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = 1
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    lattia.piirra(shader, shader_type)
    glUseProgram(0)

def init_window(asetusarvo):
    pygame.init()
    pygame.display.set_mode(asetusarvo.display, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption('jokuhomma')
    #kulli = pygame.image.load("kulli.jpg")
    #pygame.display.set_icon(kulli)
    #pygame.mixer.init()
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)
    glEnable(GL_CLIP_DISTANCE0)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

def VP(asetusarvo, shader):
    # koko maailman liikuttelu aka hahmon liikkuminen
    view_trans = pyrr.matrix44.create_from_translation(
        [asetusarvo.hahmo_koords[0], asetusarvo.hahmo_koords[1], asetusarvo.hahmo_koords[2]])
    #rot_xyz = [0 ,np.rad2deg(asetusarvo.rotator), 0] #yakselin ympari pitas menna
    rot_xyz = [-asetusarvo.x_rotation, -asetusarvo.y_rotation, 0]
    view_rot = rotate_XYZ(rot_xyz)
    view = view_rot * view_trans
    view_loc = glGetUniformLocation(shader, "view")
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    # lightpos = pyrr.matrix44.create_from_translation([0,0,3])
    lightpos = pyrr.matrix44.create_from_translation(
        pyrr.Vector3(
            [-asetusarvo.hahmo_koords[0] - 1.2, asetusarvo.hahmo_koords[1]-1, -asetusarvo.hahmo_koords[2] - 1.2]))
    Lpos = glGetUniformLocation(shader, "lightpos")
    glUniformMatrix4fv(Lpos, 1, GL_FALSE, lightpos)

    asetusarvo.Lpos = lightpos

    viewPos = pyrr.matrix44.create_from_translation(
        pyrr.Vector3([-asetusarvo.hahmo_koords[0], -asetusarvo.hahmo_koords[1], -asetusarvo.hahmo_koords[2]]))
    Vpos = glGetUniformLocation(shader, "viewPos")
    glUniformMatrix4fv(Vpos, 1, GL_FALSE, viewPos)

    # tan muuttaminen... EI
    projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1280 / 800, 0.1, 3000.0)
    proj_loc = glGetUniformLocation(shader, "projection")
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    asetusarvo.P = projection
    asetusarvo.V = view
    asetusarvo.Lpos = lightpos
    asetusarvo.Vpos = viewPos

def scale_object(obj_scale):
    scale = np.array([[obj_scale, 0.0, 0.0, 0.0],
                         [0.0, obj_scale, 0.0, 0.0],
                         [0.0, 0.0, obj_scale, 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return scale

def tubelenght(tubelen):
    scale = np.array([[0.025, 0.0, 0.0, 0.0],
                         [0.0, tubelen, 0.0, 0.0],
                         [0.0, 0.0, 0.025, 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return scale

def rengass(leveys,R):
    scale = np.array([[R, 0.0, 0.0, 0.0],
                         [0.0, leveys, 0.0, 0.0],
                         [0.0, 0.0, R, 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return scale

def rotate_XYZ(rot_xyz):
    x = (math.radians(rot_xyz[0]))
    y = (math.radians(rot_xyz[1]))
    z = (math.radians(rot_xyz[2]))
    rot_x = pyrr.Matrix44.from_x_rotation(x)
    rot_y = pyrr.Matrix44.from_y_rotation(y)
    rot_z = pyrr.Matrix44.from_z_rotation(z)
    rot = rot_x * rot_y * rot_z
    return rot

#shaderit
def normi_shader():
    vertex_shader = """
        #version 460
        layout ( location = 0 ) in vec3 position;
        layout ( location = 1 ) in vec2 tex;
        layout ( location = 2 ) in vec3 normal;
        layout ( location = 3 ) in vec3 tangent;

        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;
        uniform mat4 transform;

        const vec4 plane = vec4(0,1,0,0);

        out vec2 newTexture;
        out vec3 FragPos;
        out mat3 TBN;

        void main()
        {

            vec4 v = vec4(position,1.0f);
            gl_Position = projection * view * model * transform * v;
            FragPos = vec3(model *  transform *  v);

            //gl_ClipDistance[0] = dot((vec4(FragPos,1.0f)),plane);

            ///NORMALMAPPINGSHIT
            vec3 T = normalize(model * transform * vec4(tangent, 0.0f)).xyz;
            vec3 N = normalize(model * transform * vec4(normal, 0.0f)).xyz;
            T = normalize(T - dot(T, N) * N);
            vec3 B = normalize(cross(N, T));
            TBN = mat3(T, B, N);

            newTexture = tex;
        }
        """

    fragment_shader = """
        #version 460
        in vec2 newTexture;
        in vec3 FragPos;
        in mat3 TBN;

        uniform mat4 lightpos;
        uniform mat4 viewPos;
        uniform sampler2D samplerTexture;
        uniform sampler2D normalMap;
        uniform float luhtu;
        uniform float specularStrenght;


        vec4 Lightpos = lightpos * vec4(1.0, 1.0, 1.0, 1.0); //vec4(5.0, 10.0, 0.0, 1.0);//
        vec4 ViewPos = viewPos * vec4(1.0, 1.0, 1.0, 1.0);
        vec3 lightColor = vec3(0.9f,0.55f,0.2f);//*luhtu;

        uniform float ambientSTR;
        vec3 ambient = vec3(0.2f,0.2f,0.15f)*ambientSTR;

        vec4 Lightpos2 = vec4(0.0, 3.0, 5.0, 0.0);
        vec3 lightColor2 = vec3(1.0f,1.0f,1.0f);//*0;

        vec4 Lightpos3 = vec4(0.0, 100.0, 0.0, 1.0);
        uniform float sunlightSTR;
        //vec3 lightColor3 = vec3(0.9f,0.55f,0.2f)*sunlightSTR;
        vec3 lightColor3 = vec3(1.0f,1.0f,1.0f)*sunlightSTR;

        ///valofeidiparametreja
        float constant = 1.0;
        float linear = 0.09;
        float quadratic = 0.032;



        out vec4 outColor;
        void main()
        {


            ///NORMALS TO WORLDSPACE
            vec3 Normal = normalize(texture(normalMap, newTexture).xyz * 2.0 - 1.0);
            Normal = normalize(TBN * Normal);

            ///DIFFUSE
            vec3 norm = normalize(vec3(Normal));
            vec3 lightDir = normalize((vec3(Lightpos)) - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            vec3 lightDir2 = normalize((vec3(Lightpos2)) - FragPos);
            float diff2 = max(dot(norm, lightDir2), 0.0);
            vec3 diffuse2 = diff2 * lightColor2;

            vec3 lightDir3 = normalize((vec3(Lightpos3)) - FragPos);
            float diff3 = max(dot(norm, lightDir3), 0.0);
            vec3 diffuse3 = diff3 * lightColor3;

            ////SPECULAR
            vec3 viewDir = normalize((vec3(ViewPos))-FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrenght * spec * lightColor;

            vec3 reflectDir2 = reflect(-lightDir2, norm);
            float spec2 = pow(max(dot(viewDir, reflectDir2), 0.0), 32);
            vec3 specular2 = specularStrenght * spec2 * lightColor2;

            vec3 reflectDir3 = reflect(-lightDir3, norm);
            float spec3 = pow(max(dot(viewDir, reflectDir3), 0.0), 32);
            vec3 specular3 = specularStrenght * spec3 * lightColor3;

            //ATTENUATION
            float distance    = length(Lightpos.xyz - FragPos);
            float attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));

            float distance2    = length(Lightpos2.xyz - FragPos);
            float attenuation2 = 1.0 / (constant + linear * distance2 + quadratic * (distance2 * distance2));

            float distance3    = length(Lightpos3.xyz - FragPos);
            float attenuation3 = 1.0 / (constant + linear * distance3 + quadratic * (distance3 * distance3));

            ///VALOSUMMA
            vec3 result = (diffuse  + specular) * attenuation;
            vec3 result2 = (diffuse2 + specular2) * attenuation2;
            vec3 result3 = (diffuse3 + specular3);

            vec3 totalResult = result + result2 + result3 + ambient;

            ///TEXTUURI
            vec4 texel = texture2D(samplerTexture, newTexture);



            ///MIXAILLAAN SE FAKING FINAL PIXEL VARI ULOS
            outColor = vec4(texel) * vec4(totalResult, 1.0f);    //*vec4(newColor, 1.0f);
        }
        """

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                   OpenGL.GL.shaders.compileShader(fragment_shader,
                                                                                   GL_FRAGMENT_SHADER))
    return shader

def nolight_shader():
    vertex_shader = """
                #version 330
                layout ( location = 0 ) in vec3 position;
                layout ( location = 1 ) in vec2 tex;
                layout ( location = 2 ) in vec3 normal;
                layout ( location = 3 ) in vec3 color;

                uniform mat4 projection;
                uniform mat4 view;
                uniform mat4 model;
                uniform mat4 transform;

                out vec3 newColor;
                out vec2 newTexture;


                void main()
                {
                    vec4 v = vec4(position,1.0f);
                    vec3 FragPos = vec3(model *  transform *  v);

                    gl_Position = projection * view * model * transform * v;
                    newColor = color;
                    newTexture = tex;
                }
                """

    fragment_shader = """
                #version 330
                in vec3 newColor;
                in vec2 newTexture;

                uniform sampler2D samplerTexture;


                out vec4 outColor;
                void main()
                {
                    ///TEXTUURI
                    vec4 texel = texture2D(samplerTexture, newTexture);
                    outColor = vec4(texel);//* vec4(newColor, 1.0f);
                }
                """

    shader2 = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))
    return shader2

def terrain_shader():
    vertex_shader = """
        #version 460
        layout ( location = 0 ) in vec3 position;
        layout ( location = 1 ) in vec2 tex;
        layout ( location = 2 ) in vec3 normal;

        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;
        uniform mat4 transform;

        const vec4 plane = vec4(0,1,0,0);

        out vec2 newTexture;
        out vec3 FragPos;
        out vec3 Norm;

        void main()
        {

            vec4 v = vec4(position,1.0f);
            gl_Position = projection * view * model * transform * v;
            FragPos = vec3(model *  transform *  v);

            //gl_ClipDistance[0] = dot((vec4(FragPos,1.0f)),plane);

            Norm = normal;
            newTexture = tex;
        }
        """

    fragment_shader = """
        #version 460
        in vec2 newTexture;
        in vec3 FragPos;
        in vec3 Norm;

        uniform mat4 lightpos;
        uniform mat4 viewPos;

        uniform sampler2D blendMap;
        uniform sampler2D BGTex;
        uniform sampler2D rTex;
        uniform sampler2D gTex;
        uniform sampler2D bTex;

        uniform sampler2D BGnormalMap;
        uniform sampler2D RnormalMap;
        uniform sampler2D GnormalMap;
        uniform sampler2D BnormalMap;


        uniform float luhtu;
        uniform float specularStrenght;

        vec4 Lightpos = lightpos * vec4(1.0, 1.0, 1.0, 1.0);
        vec4 ViewPos = viewPos * vec4(1.0, 1.0, 1.0, 1.0);
        vec3 lightColor = vec3(0.9f,0.55f,0.2f);//*luhtu;
        uniform float ambientSTR;
        vec3 ambient = vec3(0.8f,0.8f,0.8f)*ambientSTR*1.2;

        vec4 Lightpos2 = vec4(1.0, 1.0, 1.0, 1.0);
        vec3 lightColor2 = vec3(1.0f,0.0f,0.0f)*0;

        vec4 Lightpos3 = vec4(0.0f, 100.0f, 0.0f, 1.0f);
        uniform float sunlightSTR;
        //vec3 lightColor3 = vec3(0.5f,0.5f,0.5f)*sunlightSTR;
        vec3 lightColor3 = vec3(1.0f,1.0f,1.0f);//sunlightSTR;

        ///valofeidiparametreja
        float constant = 1.0;
        float linear = 0.09;
        float quadratic = 0.032;



        out vec4 outColor;
        void main()
        {


            vec4 blendMapColor = texture(blendMap, newTexture);
            float BGtexAmount = 1 - (blendMapColor.r + blendMapColor.g + blendMapColor.b);
            vec2 texdensity = newTexture * 120.0;
            vec4 BGtexColor = texture(BGTex, texdensity)*BGtexAmount;
            vec4 rtexColor = texture(rTex, texdensity)*blendMapColor.r;
            vec4 gtexColor = texture(gTex, texdensity)*blendMapColor.g;
            vec4 btexColor = texture(bTex, texdensity)*blendMapColor.b;
            vec4 totalColor = BGtexColor + rtexColor + gtexColor + btexColor;


            vec4 BGnormal = texture(BGnormalMap, texdensity) * BGtexAmount;
            vec4 rnormal = texture(RnormalMap, texdensity) * blendMapColor.r;
            vec4 gnormal = texture(GnormalMap, texdensity) * blendMapColor.g;
            vec4 bnormal = texture(BnormalMap, texdensity) * blendMapColor.b;
            vec4 totalnormal = BGnormal + rnormal + gnormal + bnormal;




            // compute tangent T and bitangent B
            vec3 Q1 = dFdx(FragPos);
            vec3 Q2 = dFdy(FragPos);
            vec2 st1 = dFdx(texdensity);
            vec2 st2 = dFdy(texdensity);

            vec3 T = normalize(Q1*st2.t - Q2*st1.t);
            vec3 B = normalize(-Q1*st2.s + Q2*st1.s);
            vec3 N = normalize(Norm);

            // the transpose of texture-to-eye space matrix
            mat3 TBN = mat3(T, B, N);


            vec3 Normal = normalize(totalnormal.xyz * 2.0 - 1.0);
            Normal = (TBN * Normal);




            ///DIFFUSE
            vec3 norm = normalize(vec3(Normal));
            vec3 lightDir = normalize((vec3(Lightpos)) - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            vec3 lightDir2 = normalize((vec3(Lightpos2)) - FragPos);
            float diff2 = max(dot(norm, lightDir2), 0.0);
            vec3 diffuse2 = diff2 * lightColor2;

            vec3 lightDir3 = normalize((vec3(Lightpos3)) - FragPos);
            float diff3 = max(dot(norm, lightDir3), 0.0);
            vec3 diffuse3 = diff3 * lightColor3;

            ////SPECULAR
            vec3 viewDir = normalize((vec3(ViewPos))-FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrenght * spec * lightColor;

            vec3 reflectDir2 = reflect(-lightDir2, norm);
            float spec2 = pow(max(dot(viewDir, reflectDir2), 0.0), 32);
            vec3 specular2 = specularStrenght * spec2 * lightColor2;

            vec3 reflectDir3 = reflect(-lightDir3, norm);
            float spec3 = pow(max(dot(viewDir, reflectDir3), 0.0), 32);
            vec3 specular3 = specularStrenght * spec3 * lightColor3;

            //ATTENUATION
            float distance    = length(Lightpos.xyz - FragPos);
            float attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));

            float distance2    = length(Lightpos2.xyz - FragPos);
            float attenuation2 = 1.0 / (constant + linear * distance2 + quadratic * (distance2 * distance2));

            float distance3    = length(Lightpos3.xyz - FragPos);
            float attenuation3 = 1.0 / (constant + linear * distance3 + quadratic * (distance3 * distance3));

            ///VALOSUMMA
            vec3 result = (diffuse  + specular) * attenuation;
            vec3 result2 = (diffuse2 + specular2) * attenuation2;
            vec3 result3 = (diffuse3 + specular3);

            vec3 totalResult = result + result2 + result3 + ambient;



            ///MIXAILLAAN SE FAKING FINAL PIXEL VARI ULOS
            outColor = vec4(totalColor) * vec4(totalResult, 1.0f);
        }
        """

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                   OpenGL.GL.shaders.compileShader(fragment_shader,
                                                                                   GL_FRAGMENT_SHADER))
    return shader

def view(asetusarvo):
    init_window(asetusarvo)
    Clock = pygame.time.Clock()

    perus_shader = normi_shader()
    no_light_shader = nolight_shader()
    T_shader = terrain_shader()

    tube = Model3Dvbo("obj/tube2.obj", 1, 1, ["text/carbon.png", "nrml/laava_normal.png"], 1, 0)
    skybox = Model3Dvbo("obj/skybox.obj", 2, 3, ["text/clouds3.jpg", "nrml/laava_normal.png"], 1, 0)
    lattia = Model3Dvbo("obj/lattia.obj", 3, 5,
                        ["terrain/rock_wall.png", "terrain/rock_wall_normal.png", "terrain/rock_wall.png", "terrain/rock_wall_normal.png", \
                         "terrain/rock_wall.png", "terrain/rock_wall_normal.png", "terrain/rock_wall.png",
                         "terrain/rock_wall_normal.png", "terrain/hervantaBlend.png"], 1, 0)
                        #["terrain/BGFLOOR.jpg", "terrain/GRASSNORMAL.jpg", "terrain/SAND.jpg", "terrain/SANDNORMAL.jpg", \
                         #"terrain/grass_texture.png", "terrain/SANDNORMAL.jpg", "terrain/GRAVELL.jpg",
                         #"terrain/GRAVELNORMAL.jpg", "terrain/hervantaBlend.png"], 1, 0)



    while True:
    ############# PYGAME EVENT HANDLING #############
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    break
                if event.key == pygame.K_q:
                    asetusarvo.updown = 0.05
                if event.key == pygame.K_e:
                    asetusarvo.updown = -0.05
                if event.key == pygame.K_d:
                    asetusarvo.XZ_move_ad = +0.6
                if event.key == pygame.K_a:
                    asetusarvo.XZ_move_ad = -0.6
                if event.key == pygame.K_w:
                    asetusarvo.XZ_move_ws = +0.6
                if event.key == pygame.K_s:
                    asetusarvo.XZ_move_ws = -0.6

                if event.key == pygame.K_u:
                    if asetusarvo.euler_index < len(asetusarvo.ala_etu_euler)-1:
                        asetusarvo.euler_index += 1
                if event.key == pygame.K_i:
                    if asetusarvo.euler_index > 0:
                        asetusarvo.euler_index -= 1



                if event.key == pygame.K_p:
                    pause = True
                    while pause:
                        pygame.mouse.set_visible(True)
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    pygame.quit()
                                    return
                                if event.key == pygame.K_p:
                                    pause = False
                        pygame.time.wait(500)
                    pygame.mouse.set_visible(False)

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_q:
                    asetusarvo.updown = 0
                if event.key == pygame.K_e:
                    asetusarvo.updown = 0
                if event.key == pygame.K_d:
                    asetusarvo.XZ_move_ad = 0
                if event.key == pygame.K_a:
                    asetusarvo.XZ_move_ad = 0
                if event.key == pygame.K_w:
                    asetusarvo.XZ_move_ws = 0
                if event.key == pygame.K_s:
                    asetusarvo.XZ_move_ws = 0


            if pygame.mouse.get_pressed()[0]:
                mouse_freelook(asetusarvo)



###################################################

        #mouse_freelook(asetusarvo)
        #liikkuminen(asetusarvo)
        hahmo_move(asetusarvo)

        #print(asetusarvo.hahmo_koords)

        glEnable(GL_DEPTH_TEST)
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        shader = perus_shader  # match
        shader_type = "perus_shader"  # match
        glUseProgram(shader)
        VP(asetusarvo, shader)
        render(asetusarvo, tube, shader, shader_type)
        glUseProgram(0)

        shader = no_light_shader  # match
        shader_type = "no_light_shader"  # match
        render_nolight(asetusarvo, shader, shader_type, skybox)

        shader = T_shader  # match
        shader_type = "T_shader"  # match
        render_terrain(lattia, asetusarvo, shader, shader_type)


        pygame.display.flip()

        #vammapaskaa
        asetusarvo.mousepos_edellinen_x = pygame.mouse.get_pos()[0]
        asetusarvo.mousepos_edellinen_y = pygame.mouse.get_pos()[1]

def mouse_freelook(asetusarvo):


    mousepos_x = pygame.mouse.get_pos()[0]
    asetusarvo.y_rotation -= ((asetusarvo.mousepos_edellinen_x - mousepos_x) * asetusarvo.sens)
    asetusarvo.mousepos_edellinen_x = mousepos_x

    edellinen_x_rot = asetusarvo.x_rotation

    mousepos_y = pygame.mouse.get_pos()[1]
    if asetusarvo.invert_y:
        asetusarvo.x_rotation -= ((asetusarvo.mousepos_edellinen_y - mousepos_y) * asetusarvo.sens)
    else:
        asetusarvo.x_rotation += ((asetusarvo.mousepos_edellinen_y - mousepos_y) * asetusarvo.sens)
    asetusarvo.mousepos_edellinen_y = mousepos_y

    # mouse_freelook(asetusarvo)

    #pygame.mouse.set_pos(640, 400)

    tarkastelu = (math.sin(math.radians(asetusarvo.x_rotation)))
    if tarkastelu >= 0.98 or tarkastelu <= -0.98:
        asetusarvo.x_rotation = edellinen_x_rot

def liikkuminen(asetusarvo):
    # mouse_freelook(asetusarvo)
    etaisyys = math.sqrt(asetusarvo.hahmo_koords[2]**2 + asetusarvo.hahmo_koords[0]**2)
    if etaisyys < 1.4:
        if asetusarvo.zoom < 0:
            asetusarvo.zoom = 0
    if etaisyys > 25:
        if asetusarvo.zoom > 0:
            asetusarvo.zoom = 0

    #print(asetusarvo.zoom)

    etaisyys += asetusarvo.zoom
    asetusarvo.rotator += asetusarvo.right
    asetusarvo.hahmo_koords[1]+=asetusarvo.updown
    #asetusarvo.hahmo_koords[2] += asetusarvo.zoom
    et=math.sqrt(asetusarvo.hahmo_koords[2] ** 2 + asetusarvo.hahmo_koords[0] ** 2)
    asetusarvo.hahmo_koords[0] = -etaisyys * math.sin(asetusarvo.rotator)
    asetusarvo.hahmo_koords[2] = -etaisyys * math.cos(asetusarvo.rotator)

    #print(asetusarvo.rotator,asetusarvo.hahmo_koords[0],asetusarvo.hahmo_koords[2])

def hahmo_move(asetusarvo):
    asetusarvo.XZ_MOVE_ad += asetusarvo.XZ_move_ad
    asetusarvo.XZ_MOVE_ws += asetusarvo.XZ_move_ws
    if asetusarvo.XZ_move_ws == 0:
        asetusarvo.XZ_MOVE_ws = 0
    if asetusarvo.XZ_move_ad == 0:
        asetusarvo.XZ_MOVE_ad = 0

    if asetusarvo.XZ_MOVE_ws >= asetusarvo.Max_nopeus:
        asetusarvo.XZ_MOVE_ws = asetusarvo.Max_nopeus
    if asetusarvo.XZ_MOVE_ws <= -asetusarvo.Max_nopeus:
        asetusarvo.XZ_MOVE_ws = -asetusarvo.Max_nopeus

    if asetusarvo.XZ_MOVE_ad >= asetusarvo.Max_nopeus:
        asetusarvo.XZ_MOVE_ad = asetusarvo.Max_nopeus
    if asetusarvo.XZ_MOVE_ad <= -asetusarvo.Max_nopeus:
        asetusarvo.XZ_MOVE_ad = -asetusarvo.Max_nopeus

    x_kerroin_ws = (asetusarvo.Kiihtyvyys) * math.sin(math.radians(asetusarvo.y_rotation))
    z_kerroin_ws = -(asetusarvo.Kiihtyvyys) * math.cos(math.radians(asetusarvo.y_rotation))

    x_kerroin_ad = (asetusarvo.Kiihtyvyys) * math.sin(math.radians(asetusarvo.y_rotation + 90))
    z_kerroin_ad = -(asetusarvo.Kiihtyvyys) * math.cos(math.radians(asetusarvo.y_rotation + 90))

    asetusarvo.XZ_move_vektori = [-(asetusarvo.XZ_MOVE_ws * x_kerroin_ws) - (asetusarvo.XZ_MOVE_ad * x_kerroin_ad),
                                  -(asetusarvo.XZ_MOVE_ws * z_kerroin_ws) - (asetusarvo.XZ_MOVE_ad * z_kerroin_ad)]
    asetusarvo.hahmo_koords[0] += asetusarvo.XZ_move_vektori[0]
    asetusarvo.hahmo_koords[2] += asetusarvo.XZ_move_vektori[1]

    asetusarvo.hahmo_koords[1] += asetusarvo.updown


#computation
def trigsolver(etu,taka,kakseli):
    #c, etu taka vali
    #b, taka kakseli vali
    #a, kakseli etu vali

    a = np.linalg.norm(kakseli -etu)
    b = np.linalg.norm(kakseli -taka)
    c = np.linalg.norm(taka - etu)



    bc = np.arccos(((b**2 + c**2 - a**2)/(2*b*c))) #bc valinen kulma

    ysikymppi = np.deg2rad(90)

    c_cut = (b/np.sin(ysikymppi))*np.sin(ysikymppi-bc) #etaisyys bc

    c_cut_cord = (((taka-etu)/c)*-c_cut)+taka
    c_cut_len = np.linalg.norm(c_cut_cord-kakseli)

    kkulma = np.arcsin((etu[1]-taka[1])/c)

    return c_cut_cord,c_cut_len,kkulma, a,b

def arvot_kinematiikalle(alatuki_etu,alatuki_taka,alatuki_kakseli,ylatuki_etu,ylatuki_taka,ylatuki_kakseli):
    ala_cut = trigsolver(alatuki_etu,alatuki_taka,alatuki_kakseli)
    yla_cut = trigsolver(ylatuki_etu,ylatuki_taka,ylatuki_kakseli)
    valipala_len = np.linalg.norm(ylatuki_kakseli-alatuki_kakseli)

    tukivarsien_pituudet = [ala_cut[3],ala_cut[4],yla_cut[3],yla_cut[4]]
    #print(tukivarsien_pituudet)

    arm_up_pos = yla_cut[0]-ala_cut[0]
    #print(yla_cut[0])
    #print(ala_cut[0])
    #print(arm_up_pos)
    sift = ala_cut[0]-alatuki_etu
    #print(sift)

    alatukiRot = ala_cut[2]
    arm_up_prefi_x_xangle = np.rad2deg(yla_cut[2]) - np.rad2deg(alatukiRot)

    return ala_cut,yla_cut,valipala_len,arm_up_pos,alatukiRot,arm_up_prefi_x_xangle,sift,tukivarsien_pituudet

def endpoints_ylakulmasta(ala_cut,yla_cut,valipala_len,arm_up_pos,alatukiRot,arm_up_prefi_x_xangle,muuttuva_kulma):

    ### HAETAAN YLATUKIVARREN paatepiste MUUTTAMALLA SEN kulmaa

    # kulma alatukivarteen pyorittamalla ylatukivartta vastapaivaan... laskenta.. ja koko systeemiä myötäpäivään

    Yy = (arm_up_pos[1]*np.cos(-alatukiRot)) - (arm_up_pos[2]*np.sin(-alatukiRot))
    Yz = (arm_up_pos[1]*np.sin(-alatukiRot)) + (arm_up_pos[2]*np.cos(-alatukiRot))


    arm_up = tinyik.Actuator(["x","z",[yla_cut[1],0,0]]) #tukivarren pituus
    arm_up.angles = np.deg2rad([arm_up_prefi_x_xangle,muuttuva_kulma])
    real_up_ee = arm_up.ee + np.array([arm_up_pos[0],Yy,Yz])


    #### YLATUKIVARREN PAATEPISTEELLA ALATUKIVARREN KULMA JA VALIKAPPALEEN KULMAT
    arm=tinyik.Actuator(["z",[ala_cut[1],0,0],"x","y","z",[valipala_len,0,0]]) #alatukivarren ja valipaskan pituudet mainityssa jarjesyksessa
    arm.ee = [real_up_ee]

    YYy = (real_up_ee[1] * np.cos(alatukiRot)) - (real_up_ee[2] * np.sin(alatukiRot))
    YYz = (real_up_ee[1] * np.sin(alatukiRot)) + (real_up_ee[2] * np.cos(alatukiRot))

    YLATUKIVARSI_END = np.array([real_up_ee[0],YYy,YYz])


    aa = arm.angles[0]
    Ay = ((ala_cut[1]*np.sin(aa)) * (np.cos(alatukiRot)))
    Az = ((ala_cut[1]*np.sin(aa)) * (np.sin(alatukiRot)))


    ALATUKIVVARSI_END = np.array([ala_cut[1]*np.cos(aa),Ay,Az])


    vecc = np.array([YLATUKIVARSI_END-ALATUKIVVARSI_END])
    #print("valikappaleen pituus",np.linalg.norm(vecc)) #HYVA TARKASTUS MENEEKO KAIKKI OK

    return YLATUKIVARSI_END,ALATUKIVVARSI_END

def plotteri(yla,ala,yla_varsi,ala_varsi,plottisetteja):
    yX = []
    yY = []
    yZ = []

    aX = []
    aY = []
    aZ = []

    yvX = []
    yvY = []
    yvZ = []

    avX = []
    avY = []
    avZ = []

    for i in range(len(yla)):
        yX.append(yla[i][0])
        yY.append(yla[i][1])
        yZ.append(yla[i][2])

        aX.append(ala[i][0])
        aY.append(ala[i][1])
        aZ.append(ala[i][2])

    for o in range(len(yla_varsi)):
        yvX.append(yla_varsi[o][0])
        yvY.append(yla_varsi[o][1])
        yvZ.append(yla_varsi[o][2])

        avX.append(ala_varsi[o][0])
        avY.append(ala_varsi[o][1])
        avZ.append(ala_varsi[o][2])

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(yX,yY,yZ, c="r", zdir="z")
    ax.plot(aX, aY, aZ, c="b", zdir="z")
    ax.plot(yvX, yvY, yvZ, c="g", zdir="z")
    ax.plot(avX, avY, avZ, c="g", zdir="z")
    ax.set_xlim3d(0,1)
    ax.set_ylim3d(0,1)
    ax.set_zlim3d(0,1)

    ax.set_xlabel("x")
    ax.set_ylabel("y jousto(m)")
    ax.set_zlabel("z")

    plt.figure(2)

    plt.subplot(221)
    plt.plot(plottisetteja[0],plottisetteja[3],"r")
    plt.ylabel("jousto(m)")
    plt.xlabel("camber(deg)")
    #plt.title('delta camber')
    plt.grid(True)

    plt.subplot(222)
    plt.plot(plottisetteja[1], plottisetteja[3], "r")
    plt.ylabel("jousto(m)")
    plt.xlabel("caster(deg)")
    #plt.title('caster')
    plt.grid(True)

    plt.subplot(223)
    plt.plot(plottisetteja[2], plottisetteja[3], "r")
    plt.ylabel("jousto(m)")
    plt.xlabel("raideleveys(m)")
    #plt.title('raideleveys')
    plt.grid(True)




    plt.show()

def compute(asetusarvo): #tarvii sit siirrettyna noi inputtina
    alatuki_etu = asetusarvo.alatuki_etu #etutakaxsamassa tasossa siftaa jotenkin oudosti jos ei o
    alatuki_taka = asetusarvo.alatuki_taka#etutakaxsamassa tasossa
    alatuki_kakseli = asetusarvo.alatuki_kakseli_init

    ylatuki_etu = asetusarvo.ylatuki_etu#etutakaxsamassa tasossa
    ylatuki_taka = asetusarvo.ylatuki_taka#etutakaxsamassa tasossa
    ylatuki_kakseli = asetusarvo.ylatuki_kakseli_init

    renkaan_keski = np.array([0.8,0.1,0.7])
    renkaan_R = 0.3

    init_ang_n_vec = init_angles(alatuki_kakseli, ylatuki_kakseli, renkaan_keski, renkaan_R)
    #print(init_ang_n_vec)

    arvoja = arvot_kinematiikalle(alatuki_etu, alatuki_taka, alatuki_kakseli, ylatuki_etu, ylatuki_taka, ylatuki_kakseli)

    #print(arvoja)
    yla = []
    ala = []

    for muuttuva_kulma in range(-30,70,2):
        yla_ala = endpoints_ylakulmasta(arvoja[0],arvoja[1],arvoja[2],arvoja[3],arvoja[4],arvoja[5],muuttuva_kulma)
        yla.append(yla_ala[0]+arvoja[6]) #arvoja6 on sifti
        ala.append(yla_ala[1]+arvoja[6])

    plottisetteja = rengas_tiehen(init_ang_n_vec,yla,ala,renkaan_keski,renkaan_R)


    ala_varsi = [alatuki_etu-alatuki_etu,alatuki_kakseli-alatuki_etu,alatuki_taka-alatuki_etu]
    yla_varsi = [ylatuki_etu-alatuki_etu,ylatuki_kakseli-alatuki_etu,ylatuki_taka-alatuki_etu]


    if asetusarvo.PLOT:
        plotteri(yla,ala,yla_varsi,ala_varsi,plottisetteja)

    else:
        asetusarvo.ala_etu_euler = euler_yla_ala(ala, ala_varsi[0])
        asetusarvo.ala_taka_euler = euler_yla_ala(ala, ala_varsi[2])

        asetusarvo.yla_etu_euler = euler_yla_ala(yla, yla_varsi[0])
        asetusarvo.yla_taka_euler = euler_yla_ala(yla, yla_varsi[2])

        tukivarsien_pituudet = arvoja[7] #palautettava for 3d alaetu, alataka, ylaetu, ylataka pituudet
        #print(tukivarsien_pituudet)
        #print(arvoja[2]) #valipalalen

        asetusarvo.alatuki_etu_len = tukivarsien_pituudet[0]
        asetusarvo.alatuki_taka_len = tukivarsien_pituudet[1]
        asetusarvo.ylatuki_etu_len = tukivarsien_pituudet[2]
        asetusarvo.ylatuki_taka_len = tukivarsien_pituudet[3]
        asetusarvo.valipala_len = arvoja[2]

        asetusarvo.rengas_keski = plottisetteja[4]


        #print(ala) #palautettava.. sidospaikka valipalalle
        #print(asetusarvo.ala_etu_euler) #palautettava for 3d
        #print(asetusarvo.ala_taka_euler) #palautettava for 3d

        #print(asetusarvo.yla_etu_euler) #palautettava for 3d
        #print(asetusarvo.yla_taka_euler) #palautettava for 3d

        asetusarvo.jousto = plottisetteja[3]
        asetusarvo.alatuki_kakseli = ala
        asetusarvo.valipala_euler = plottisetteja[5]

        asetusarvo.rengastie = plottisetteja[6]
        #tarvii vial valipala eulerifunktion

        view(asetusarvo)

def euler_yla_ala(loose_end, kiinnitys):
    eulers = []
    tappi = np.array([0,1,0])
    for osa in loose_end:
        vec = osa-kiinnitys
        lenvec = np.linalg.norm(vec)
        v_hat = vec / lenvec
        arm = tinyik.Actuator(["x","y","z", tappi])  # alatukivarren ja valipaskan pituudet mainityssa jarjesyksessa
        arm.ee = [v_hat]
        eulers.append(-np.rad2deg(arm.angles)) #negaatio koska käänteinen 3d maailma
    return eulers

def init_angles(alatuki_kakseli,ylatuki_kakseli,renkaan_keski,renkaan_R):

    tie_kosketus = renkaan_keski - np.array([0,renkaan_R,0])

    keski_tappi_len = np.linalg.norm(renkaan_keski-alatuki_kakseli)
    ala2tielen = np.linalg.norm(tie_kosketus-alatuki_kakseli)
    valipala_len = np.linalg.norm(ylatuki_kakseli-alatuki_kakseli)

    #print(valipala_len)

    renkaankeski_init = tinyik.Actuator(["x","y","z", [keski_tappi_len, 0, 0]])  #renkaan keski alakaantopaskan suhteen
    renkaankeski_init.ee = renkaan_keski-alatuki_kakseli
    ala2keski_ang = renkaankeski_init.angles

    rengastie_init = tinyik.Actuator(["x", "y", "z", [ala2tielen, 0, 0]])  # renkaan keski alakaantopaskan suhteen
    rengastie_init.ee = tie_kosketus - alatuki_kakseli
    ala2tie_ang = rengastie_init.angles


    valipala_init = tinyik.Actuator(["x", "y", "z", [valipala_len, 0, 0]])#valipalan init kulmat
    valipala_init.ee = ylatuki_kakseli - alatuki_kakseli
    valipala_ang = valipala_init.angles

    return ala2keski_ang, ala2tie_ang, valipala_ang,valipala_len,keski_tappi_len,ala2tielen

def rengas_tiehen(init_ang_n_vec,yla,ala,renkaan_keski,renkaan_R):
    VALIPALA_EULER = []
    DELTA_ANG = []
    RENGAS_KESKI = []
    RENGAS_TIE = []
    DELTA_CAMBER = []
    CASTER = []
    DELTA_RAIDELEVEYS = []
    JOUSTO = []

    init_kosketus = renkaan_keski-np.array([0,renkaan_R,0])


    for i in range(len(yla)):
        valipala = tinyik.Actuator(["x", "y", "z", [init_ang_n_vec[3], 0, 0]])
        valipala.ee = yla[i]-ala[i]
        valipala_eul = valipala.angles
        delta_ang = valipala_eul - init_ang_n_vec[2]
        DELTA_ANG.append(delta_ang)
        DELTA_CAMBER.append(np.rad2deg(-delta_ang[2]))
        VALIPALA_EULER.append(valipala_eul)
        CASTER.append(np.rad2deg(valipala_eul[0]))

        rengaskeski = tinyik.Actuator(["x", "y", "z", [init_ang_n_vec[4], 0, 0]])
        rengaskeski.angles = init_ang_n_vec[0] + delta_ang
        rengaskeski_pos = rengaskeski.ee + ala[i]
        RENGAS_KESKI.append(rengaskeski_pos)

        rengastie = tinyik.Actuator(["x", "y", "z", [init_ang_n_vec[5], 0, 0]])
        rengastie.angles = init_ang_n_vec[1] + delta_ang
        rengastie_pos = rengastie.ee + ala[i]
        RENGAS_TIE.append(rengastie_pos)

        delta_lev = init_kosketus - rengastie_pos
        DELTA_RAIDELEVEYS.append(-delta_lev[0])
        JOUSTO.append(-delta_lev[1])

    print(VALIPALA_EULER)

    return DELTA_CAMBER,CASTER,DELTA_RAIDELEVEYS,JOUSTO,RENGAS_KESKI,VALIPALA_EULER,RENGAS_TIE

def main(asetusarvo):
    root = tk.Tk()
    root.title("GUI :D vittu tk GUI")

    def comp_plot():
        asetusarvo.PLOT = True
        compute(asetusarvo)

    def katselu():
        asetusarvo.PLOT = False
        compute(asetusarvo)

    button_1 = tk.Button(root, text="3d view", command=katselu, fg="red")
    button_1.grid(row=1, column=0)
    button_2 = tk.Button(root, text="plotit", command=comp_plot,fg="red")
    button_2.grid(row=1, column=1, sticky=tk.W)


    root.mainloop()

if __name__ == "__main__":
    asetusarvo = ASETUSARVOJA()
    main(asetusarvo)