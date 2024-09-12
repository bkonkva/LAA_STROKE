DROP TABLE IF EXISTS dflt.Demographics_ND

--Get Race Information
;with RaceCTE as ( --4311360
select *, row_number() over (partition by patientsid order by patientsid, patientRaceSID desc) rownum FROM [ORD_Kicska_202402007D].Src.[PatSub_PatientRace] as Race 
) select *  INTO #Race from RaceCTE where rownum = 1



--Get Ethnicity Information



;WITH EthnicityCTE as ( --select * FROM  [ORD_Kicska_202402007D].Src.[PatSub_PatientEthnicity] 
select *, row_number() over (partition by patientsid order by patientsid,PatientEthnicitySID desc) rownum FROM [ORD_Kicska_202402007D].Src.[PatSub_PatientEthnicity] as Ethnicity 
) 
select * INTO #Ethnicity from EthnicityCTE where rownum = 1


--Build Demographics Table

SELECT DISTINCT Patient.[PatientSID]
      ,[PatientName]
	  ,Patient.Sta3n
	  ,[PatientLastName]
      ,[PatientFirstName]
      ,[TestPatientFlag]
      ,[CDWPossibleTestPatientFlag]
      ,[VeteranFlag]
      ,[PatientType]
	  ,Results.[PatientICN]
	  ,[BirthDateTime]
	--  , Results.ExamDateTime
      ,[DeceasedFlag]
      ,Patient.[Gender]
      ,[Religion]
      ,[MaritalStatus]
      ,[MeansTestStatus]
      ,[PeriodOfService]
	  ,Race.Race
	  , Ethnicity.Ethnicity
	   ,[FederalAgencySID]
      ,[FilipinoVeteranCode]
	  INTO dflt.Demographics_ND --select count(distinct Results.PatientSID) --23733
  FROM 
  [ORD_Kicska_202402007D].Dflt.ComorditiesTable as Results
  LEFT OUTER JOIN [ORD_Kicska_202402007D].[Src].[SPatient_SPatient] as Patient
  on Patient.PatientSID = Results.PatientSID 
  -- select * FROM [ORD_Kicska_202402007D].Src.[PatSub_PatientRace] as Race where patientsid = 758701
  LEFT OUTER JOIN #Race Race
  --[ORD_Kicska_202402007D].Src.[PatSub_PatientRace] as Race
  on Patient.PatientSID = isnull(Race.PatientSID,0)
  
  LEFT OUTER JOIN 
  --select top 10 * FROM
  #Ethnicity as Ethnicity
  on Patient.PatientSID = isnull(Ethnicity.PatientSID,0)
 